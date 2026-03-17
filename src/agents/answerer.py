from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.agents.retriever import format_chunks_for_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# src/agents/answerer.py — substitua _SYSTEM_PROMPT e _HUMAN_TEMPLATE

_SYSTEM_PROMPT = """\
Você é um assistente acadêmico da UFCG. Sua tarefa é responder perguntas usando os trechos de documentos fornecidos.

Formato obrigatório da resposta:
1. Escreva a resposta completa em português.
2. Ao final de cada frase com informação factual, adicione (Fonte: nome_do_arquivo, página N).
3. Termine com uma linha "Referências:" seguida das fontes usadas.
4. Nunca invente informações. Se não souber, diga que não encontrou nos documentos.
"""

_HUMAN_TEMPLATE = """\
Pergunta: {query}

Documentos disponíveis:
{evidence}

Responda à pergunta usando apenas as informações dos documentos acima.
"""

_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=_SYSTEM_PROMPT),
    HumanMessage(content=_HUMAN_TEMPLATE),
])

# Prompt alternativo para quando não há chunks suficientes
_NO_EVIDENCE_TEMPLATE = """\
A pergunta foi: "{query}"

Não foram encontrados trechos relevantes nos documentos indexados.
Informe ao usuário de forma clara que não há informações disponíveis sobre
esse tópico nos documentos da UFCG indexados, e sugira que consulte o DAA ou o SIGAA.
Seja breve (2–3 frases).
"""


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def build_answer(
    query: str,
    chunks: list[dict],
    llm: BaseChatModel,
    safety_message: str = "",
) -> str:

    if not query:
        return "Não recebi uma pergunta para responder."

    if not chunks:
        return _answer_no_evidence(query, llm, safety_message)

    # Limita para 3 chunks e 1500 chars para caber no contexto do 3B
    evidence = format_chunks_for_prompt(chunks[:3], max_chars=1500)

    try:
        import traceback
        from langchain_core.messages import SystemMessage, HumanMessage

        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_HUMAN_TEMPLATE.format(
                query=query,
                evidence=evidence,
            )),
        ])
        answer = response.content.strip()
        logger.info("[answerer] resposta gerada (%d chars)", len(answer))

        if not answer:
            logger.error("[answerer] LLM retornou resposta vazia")
            return _answer_no_evidence(query, llm, safety_message)

    except Exception as exc:
        logger.error("[answerer] ERRO:\n%s", traceback.format_exc())
        answer = f"Erro ao gerar resposta: {exc}"

    answer = _ensure_references_section(answer, chunks)
    return answer

# ---------------------------------------------------------------------------
# Resposta para ausência de evidências
# ---------------------------------------------------------------------------

def _answer_no_evidence(query: str, llm: BaseChatModel, safety_message: str) -> str:
    """Gera uma resposta curta informando que não há evidências disponíveis."""
    try:
        messages = [
            SystemMessage(content="Você é um assistente acadêmico da UFCG."),
            HumanMessage(content=_NO_EVIDENCE_TEMPLATE.format(query=query)),
        ]
        response = llm.invoke(messages)
        answer = response.content.strip()
    except Exception:
        answer = (
            f"Não encontrei informações sobre \"{query}\" nos documentos indexados. "
            "Consulte o DAA ou o SIGAA para obter essa informação."
        )

    if safety_message and safety_message not in answer:
        answer += f"\n\n---\n> ℹ️ {safety_message}"

    return answer


# ---------------------------------------------------------------------------
# Garantia da seção de referências
# ---------------------------------------------------------------------------

def _ensure_references_section(answer: str, chunks: list[dict]) -> str:
    """
    Se o LLM não incluiu a seção de Referências, adiciona automaticamente
    com base nos chunks disponíveis.
    """
    if "## Referências" in answer or "## referencias" in answer.lower():
        return answer

    # Deduplica fontes
    seen: set[str] = set()
    refs: list[str] = []
    for chunk in chunks:
        source = chunk.get("source", "")
        page = chunk.get("page")
        section = chunk.get("section", "")

        key = f"{source}:{page}"
        if key in seen:
            continue
        seen.add(key)

        parts = [f"**{source}**"]
        if page:
            parts.append(f"pág. {page}")
        if section:
            parts.append(f'seção: "{section}"')
        refs.append("- " + ", ".join(parts))

    if refs:
        refs_section = "\n\n## Referências\n" + "\n".join(refs)
        answer += refs_section

    return answer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="qwen2.5:3b", temperature=0.3)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "Quais são os pré-requisitos de Redes de Computadores?"

    chunks = [
        {
            "text": "COMP3501 - Redes de Computadores. Pré-requisitos: COMP2401 (Sistemas Operacionais) e COMP2201 (Fundamentos de Redes). Carga horária: 60h. Período recomendado: 6º.",
            "source": "fluxograma_cc.pdf",
            "page": 5,
            "section": "Grade Curricular — 6º Período",
            "score": 0.92,
            "excerpt": "COMP3501 - Redes de Computadores. Pré-requisitos: COMP2401 e COMP2201.",
        },
        {
            "text": "Art. 23 — Para efeito de matrícula, considera-se pré-requisito a disciplina que deve ter sido cursada com aprovação antes da disciplina subsequente.",
            "source": "regulamento_graduacao.pdf",
            "page": 12,
            "section": "Capítulo IV — Matrícula",
            "score": 0.78,
            "excerpt": "Art. 23 — Para efeito de matrícula...",
        },
    ]

    print(f"Query: {query}\n{'='*60}")
    answer = build_answer(query, chunks, llm)
    print(answer)