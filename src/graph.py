# src/graph.py
# LangGraph StateGraph — Assistente de Matrícula UFCG
# Agentes: Supervisor → Retriever → Safety → Self-Check → Answerer
#          Supervisor → Automation Agent (pré-req, conflito, trilha)
#          Supervisor → Recusa

from __future__ import annotations

import json
import logging
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Importações internas do projeto
# ---------------------------------------------------------------------------
from src.agents.answerer import build_answer
from src.agents.automation import run_automation
from src.agents.retriever import retrieve_chunks
from src.agents.safety import apply_safety
from src.agents.self_check import self_check
from src.agents.supervisor import classify_intent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Estado compartilhado entre todos os nós
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Histórico de mensagens (append-only via add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Última pergunta do usuário (extraída no nó de entrada)
    query: str

    # Intent classificada pelo Supervisor: "qa", "automation", "refuse"
    intent: Literal["qa", "automation", "refuse"]

    # Sub-intent de automação: "prereq", "schedule", "trail", ou None
    automation_type: Literal["prereq", "schedule", "trail"] | None

    # Chunks recuperados pelo Retriever [{text, source, page, score}]
    retrieved_chunks: list[dict]

    # Rascunho gerado pelo Answerer antes do Self-Check
    draft_answer: str

    # Score do Self-Check (0.0–1.0). Abaixo de 0.7 → re-busca
    self_check_score: float

    # Quantas vezes o Self-Check já pediu re-busca (máx 1)
    retry_count: int

    # Resposta final formatada com citações
    final_answer: str

    # Resultado da automação (dict livre, serializado para string na resposta)
    automation_result: dict | None

    # Flag: Safety bloqueou a resposta?
    safety_blocked: bool

    # Mensagem de aviso do Safety (disclaimer ou recusa parcial)
    safety_message: str


# ---------------------------------------------------------------------------
# LLM compartilhado (Qwen 2.5 via Ollama)
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.0) -> ChatOllama:
    """
    Retorna o LLM Qwen 2.5.
    temperature=0 nos agentes de classificação/verificação para máxima consistência.
    temperature=0.3 no Answerer para respostas mais naturais.
    """
    return ChatOllama(
        model="qwen2.5:3b",  # troque por qwen2.5:14b se tiver GPU suficiente
        temperature=temperature,
        num_ctx=8192,          # contexto estendido para RAG
    )


# ---------------------------------------------------------------------------
# Nó 0 — Entrada: extrai a query da última mensagem humana
# ---------------------------------------------------------------------------

def node_entry(state: AgentState) -> dict:
    """Extrai a query mais recente do histórico de mensagens."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    query = last_human.content if last_human else ""
    logger.info("[entry] query=%r", query)
    return {
        "query": query,
        "retry_count": state.get("retry_count", 0),
        "self_check_score": 0.0,
        "safety_blocked": False,
        "safety_message": "",
        "automation_result": None,
        "draft_answer": "",
        "final_answer": "",
    }


# ---------------------------------------------------------------------------
# Nó 1 — Supervisor: classifica a intent
# ---------------------------------------------------------------------------

def node_supervisor(state: AgentState) -> dict:
    """
    Classifica a query em uma de três rotas:
      - "qa"         → fluxo RAG completo
      - "automation" → fluxo de automação (pré-req, conflito, trilha)
      - "refuse"     → pergunta fora do escopo ou sem dados suficientes
    """
    llm = get_llm(temperature=0.0)
    intent, automation_type = classify_intent(state["query"], llm)
    logger.info("[supervisor] intent=%s  automation_type=%s", intent, automation_type)
    return {"intent": intent, "automation_type": automation_type}


# ---------------------------------------------------------------------------
# Nó 2a — Retriever: busca semântica no FAISS
# ---------------------------------------------------------------------------

def node_retriever(state: AgentState) -> dict:
    """
    Recupera os top-k chunks mais relevantes do corpus UFCG.
    Usa FAISS + embeddings bge-m3 (HuggingFace).
    """
    chunks = retrieve_chunks(
        query=state["query"],
        top_k=6,
        score_threshold=0.45,   # descarta chunks muito distantes semanticamente
    )
    logger.info("[retriever] %d chunks recuperados", len(chunks))
    return {"retrieved_chunks": chunks}


# ---------------------------------------------------------------------------
# Nó 2b — Automation: executa a automação solicitada
# ---------------------------------------------------------------------------

def node_automation(state: AgentState) -> dict:
    """
    Executa uma das três automações:
      - prereq:   verifica pré-requisitos de uma lista de disciplinas
      - schedule: detecta conflitos de horário entre disciplinas
      - trail:    gera trilha de estudos recursiva para uma disciplina-alvo
    O Automation Agent usa o MCP docstore como ferramenta de consulta.
    """
    result = run_automation(
        automation_type=state["automation_type"],
        query=state["query"],
        # Passa chunks se o retriever já rodou antes (pode ser chamado em conjunto)
        context_chunks=state.get("retrieved_chunks", []),
    )
    logger.info("[automation] type=%s  result_keys=%s", state["automation_type"], list(result.keys()))
    return {"automation_result": result}


# ---------------------------------------------------------------------------
# Nó 3 — Safety / Policy: adiciona disclaimers e bloqueia se necessário
# ---------------------------------------------------------------------------

def node_safety(state: AgentState) -> dict:
    """
    Verifica se a resposta (ou automação) viola a política de escopo.
    - Adiciona disclaimers quando relevante (ex.: "consulte o DAA")
    - Bloqueia respostas que aconselham fora do escopo acadêmico
    """
    blocked, message = apply_safety(
        query=state["query"],
        draft=state.get("draft_answer", ""),
        automation_result=state.get("automation_result"),
    )
    if blocked:
        logger.warning("[safety] resposta bloqueada: %s", message)
    return {"safety_blocked": blocked, "safety_message": message}


# ---------------------------------------------------------------------------
# Nó 4 — Answerer: gera o rascunho com citações
# ---------------------------------------------------------------------------

def node_answerer(state: AgentState) -> dict:
    """
    Gera a resposta formatada em markdown com:
      - Resposta principal
      - Citações inline: [Fonte: <doc>, pág. <n>, trecho: "..."]
      - Seção "Referências" ao final
    """
    llm = get_llm(temperature=0.3)
    draft = build_answer(
        query=state["query"],
        chunks=state["retrieved_chunks"],
        llm=llm,
        safety_message=state.get("safety_message", ""),
    )
    logger.info("[answerer] rascunho gerado (%d chars)", len(draft))
    return {"draft_answer": draft}


# ---------------------------------------------------------------------------
# Nó 5 — Self-Check: valida se as afirmações têm suporte nos chunks
# ---------------------------------------------------------------------------

def node_self_check(state: AgentState) -> dict:
    """
    Self-RAG: pede ao LLM para avaliar se cada afirmação do draft
    está suportada pelos chunks recuperados.
    Retorna um score 0.0–1.0 e registra para a edge condicional.
    """
    llm = get_llm(temperature=0.0)
    score = self_check(
        draft=state["draft_answer"],
        chunks=state["retrieved_chunks"],
        llm=llm,
    )
    logger.info("[self_check] score=%.2f  retry_count=%d", score, state.get("retry_count", 0))
    return {"self_check_score": score}


# ---------------------------------------------------------------------------
# Nó 6 — Resposta final: formata para o usuário
# ---------------------------------------------------------------------------

def node_final_answer(state: AgentState) -> dict:
    """
    Consolida o rascunho (ou resultado de automação) na resposta final.
    Injeta o disclaimer do Safety se houver.
    """
    if state.get("safety_blocked"):
        answer = (
            f"⚠️ Não consigo responder a essa pergunta com as informações disponíveis.\n\n"
            f"{state['safety_message']}"
        )
    elif state.get("automation_result"):
        answer = _format_automation_answer(
            state["automation_result"],
            state.get("safety_message", ""),
        )
    else:
        disclaimer = f"\n\n---\n> ℹ️ {state['safety_message']}" if state.get("safety_message") else ""
        answer = state["draft_answer"] + disclaimer

    logger.info("[final_answer] %d chars", len(answer))
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Nó 7 — Recusa
# ---------------------------------------------------------------------------

def node_refuse(state: AgentState) -> dict:
    """Resposta padrão quando o Supervisor decide que está fora do escopo."""
    answer = (
        "Desculpe, não consigo responder a essa pergunta com as informações "
        "disponíveis sobre a UFCG. Por favor, consulte o DAA ou o SIGAA diretamente."
    )
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Edges condicionais
# ---------------------------------------------------------------------------

def route_supervisor(state: AgentState) -> str:
    """Decide qual ramo seguir após o Supervisor."""
    return state["intent"]  # "qa", "automation", ou "refuse"


def route_self_check(state: AgentState) -> str:
    """
    Após o Self-Check:
      - score >= 0.7 → resposta aceita → final_answer
      - score < 0.7 e retry_count == 0 → re-busca (volta ao retriever)
      - score < 0.7 e retry_count >= 1 → recusa (evita loop infinito)
    """
    score = state["self_check_score"]
    retries = state.get("retry_count", 0)

    if score >= 0.7:
        return "accept"
    if retries < 1:
        return "retry"
    return "refuse"


def route_safety(state: AgentState) -> str:
    """Se o Safety bloqueou, vai direto para final_answer (que formata o bloqueio)."""
    return "blocked" if state.get("safety_blocked") else "ok"


# ---------------------------------------------------------------------------
# Construção do grafo
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # --- Registra nós ---
    # IMPORTANTE: nomes de nós NÃO podem colidir com chaves do AgentState.
    # Por isso usamos prefixo "node_" nos nós cujos nomes conflitam
    # com campos do estado (final_answer, automation_result, safety_message…).
    graph.add_node("node_entry",        node_entry)
    graph.add_node("node_supervisor",   node_supervisor)
    graph.add_node("node_retriever",    node_retriever)
    graph.add_node("node_automation",   node_automation)
    graph.add_node("node_safety",       node_safety)
    graph.add_node("node_answerer",     node_answerer)
    graph.add_node("node_self_check",   node_self_check)
    graph.add_node("node_final_answer", node_final_answer)
    graph.add_node("node_refuse",       node_refuse)

    # --- Edges fixas ---
    graph.add_edge(START,             "node_entry")
    graph.add_edge("node_entry",      "node_supervisor")

    # Supervisor → rota condicional
    graph.add_conditional_edges(
        "node_supervisor",
        route_supervisor,
        {
            "qa":         "node_retriever",
            "automation": "node_retriever",   # automação também consulta RAG
            "refuse":     "node_refuse",
        },
    )

    # Após retriever: automation vai pro nó de automação, qa vai pro safety direto
    graph.add_conditional_edges(
        "node_retriever",
        lambda s: "node_automation" if s["intent"] == "automation" else "node_safety",
        {
            "node_automation": "node_automation",
            "node_safety":     "node_safety",
        },
    )

    # Automação → safety (para disclaimers antes da resposta final)
    graph.add_edge("node_automation", "node_safety")

    # Safety → condicional
    graph.add_conditional_edges(
        "node_safety",
        route_safety,
        {
            "blocked": "node_final_answer",  # bloqueia sem passar pelo answerer
            "ok":      "node_answerer",
        },
    )

    # Answerer → self_check
    graph.add_edge("node_answerer", "node_self_check")

    # Self-Check → condicional
    graph.add_conditional_edges(
        "node_self_check",
        route_self_check,
        {
            "accept": "node_final_answer",
            "retry":  "node_retriever",     # re-busca com query ampliada (1x)
            "refuse": "node_refuse",
        },
    )

    # Fins
    graph.add_edge("node_final_answer", END)
    graph.add_edge("node_refuse",       END)

    return graph


# ---------------------------------------------------------------------------
# Compilação do grafo (singleton para uso no app)
# ---------------------------------------------------------------------------

_compiled_graph = None

def get_compiled_graph():
    """Retorna o grafo compilado (singleton). Thread-safe para Streamlit."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
        logger.info("[graph] grafo compilado com sucesso")
    return _compiled_graph


# ---------------------------------------------------------------------------
# Função principal de invocação (usada pelo Streamlit/Gradio)
# ---------------------------------------------------------------------------

def run_agent(query: str, history: list[BaseMessage] | None = None) -> str:
    """
    Ponto de entrada principal.

    Args:
        query:   Pergunta do usuário.
        history: Lista de mensagens anteriores (para manter contexto).

    Returns:
        Resposta final em markdown com citações.
    """
    graph = get_compiled_graph()

    # Monta o estado inicial
    messages = list(history or []) + [HumanMessage(content=query)]
    initial_state: AgentState = {
        "messages":         messages,
        "query":            "",
        "intent":           "qa",
        "automation_type":  None,
        "retrieved_chunks": [],
        "draft_answer":     "",
        "self_check_score": 0.0,
        "retry_count":      0,
        "final_answer":     "",
        "automation_result": None,
        "safety_blocked":   False,
        "safety_message":   "",
    }

    result = graph.invoke(initial_state)
    return result["final_answer"]


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _format_automation_answer(result: dict, safety_message: str) -> str:
    """
    Formata o resultado de uma automação para markdown legível.

    result deve conter:
      - "type": "prereq" | "schedule" | "trail"
      - "summary": string com o resultado principal
      - "details": lista de dicts com detalhes
      - "sources": lista de citações [{source, page, excerpt}]
    """
    atype = result.get("type", "")
    summary = result.get("summary", "")
    details = result.get("details", [])
    sources = result.get("sources", [])

    # Cabeçalho por tipo
    headers = {
        "prereq":   "## ✅ Verificação de Pré-Requisitos",
        "schedule": "## 🗓️ Verificação de Conflitos de Horário",
        "trail":    "## 📚 Trilha de Estudos Sugerida",
    }
    header = headers.get(atype, "## Resultado da Automação")

    lines = [header, "", summary, ""]

    if details:
        lines.append("### Detalhes")
        for item in details:
            lines.append(f"- {item}")
        lines.append("")

    if sources:
        lines.append("### Fontes")
        for src in sources:
            lines.append(
                f"- **{src.get('source', '?')}** "
                f"(pág. {src.get('page', '?')}): "
                f"*\"{src.get('excerpt', '')}\"*"
            )
        lines.append("")

    if safety_message:
        lines += ["---", f"> ℹ️ {safety_message}"]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execução direta para testes rápidos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    queries = [
        "Quais são os pré-requisitos de Redes de Computadores na UFCG?",
        "Verifique se COMP1001, COMP2003 e MAT1001 têm conflito de horário no período 2025.1",
        "Quero cursar Sistemas Operacionais, o que devo estudar antes?",
        "Qual é o horário da cantina?",   # deve ser recusado
    ]

    query = sys.argv[1] if len(sys.argv) > 1 else queries[0]
    print(f"\nQuery: {query}\n{'='*60}")
    answer = run_agent(query)
    print(answer)