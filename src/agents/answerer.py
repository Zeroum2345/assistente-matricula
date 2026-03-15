# src/agents/answerer.py
# Answerer / Writer Agent — gera a resposta final com citações obrigatórias
#
# Responsabilidades:
#   1. Receber a query, os chunks recuperados e o safety_message
#   2. Construir um prompt rico com as evidências formatadas
#   3. Pedir ao LLM para gerar resposta em markdown com citações inline
#   4. Garantir que toda afirmação factual tenha [Fonte: X] correspondente

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

_SYSTEM_PROMPT = """\
Você é um assistente acadêmico especializado na UFCG (Universidade Federal de
Campina Grande). Responde perguntas sobre regulamentos, disciplinas, pré-requisitos,
horários e normas acadêmicas usando EXCLUSIVAMENTE os trechos de documentos fornecidos.

=== REGRAS OBRIGATÓRIAS ===

1. CITAÇÕES: Toda afirmação factual deve ter uma citação inline no formato:
   [Fonte: <nome_do_arquivo>, pág. <N>]
   Se não houver número de página, use:
   [Fonte: <nome_do_arquivo>]

2. APENAS EVIDÊNCIAS: Não inclua informações que não estejam nos trechos fornecidos.
   Se a informação não estiver disponível, diga explicitamente.

3. FORMATO DA RESPOSTA:
   - Responda em português brasileiro, tom formal mas acessível.
   - Use markdown: **negrito** para termos importantes, listas quando aplicável.
   - Inclua uma seção "## Referências" ao final listando todas as fontes citadas.
   - Seja conciso: prefira 3–5 parágrafos a respostas muito longas.

4. SEM INVENÇÃO: Nunca complete lacunas com suposições. Se os trechos forem
   insuficientes, declare: "Não encontrei informações suficientes sobre [X] nos
   documentos disponíveis."

=== EXEMPLO DE CITAÇÃO CORRETA ===
Para cursar **Redes de Computadores**, o aluno deve ter sido aprovado em
**Sistemas Operacionais** e **Fundamentos de Redes**
[Fonte: fluxograma_cc.pdf, pág. 5].

## Referências
- fluxograma_cc.pdf, pág. 5 — Pré-requisitos de COMP3501
"""

_HUMAN_TEMPLATE = """\
=== PERGUNTA ===
{query}

=== TRECHOS DE DOCUMENTOS (use apenas estas informações) ===
{evidence}

Responda à pergunta acima com citações obrigatórias em cada afirmação factual.
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
    """
    Gera a resposta formatada com citações.

    Args:
        query:          Pergunta do usuário.
        chunks:         Chunks recuperados pelo Retriever.
        llm:            LLM (ChatOllama com temperature ~0.3).
        safety_message: Disclaimer do Safety Agent (injetado ao final).

    Returns:
        Resposta em markdown com citações inline e seção de referências.
    """
    if not query:
        return "Não recebi uma pergunta para responder."

    # Sem evidências suficientes → resposta de "não encontrei"
    if not chunks:
        logger.warning("[answerer] sem chunks — gerando resposta de ausência de evidências")
        return _answer_no_evidence(query, llm, safety_message)

    evidence = format_chunks_for_prompt(chunks, max_chars=3000)

    try:
        messages = _PROMPT.format_messages(query=query, evidence=evidence)
        response = llm.invoke(messages)
        answer = response.content.strip()
        logger.info("[answerer] resposta gerada (%d chars)", len(answer))

    except Exception as exc:
        logger.error("[answerer] erro ao chamar LLM: %s", exc)
        answer = (
            "Desculpe, ocorreu um erro ao gerar a resposta. "
            "Por favor, tente novamente ou consulte o DAA diretamente."
        )

    # Garante que a seção de Referências existe
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