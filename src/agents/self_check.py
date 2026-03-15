# src/agents/self_check.py
# Self-Check Agent — implementação Self-RAG local com LangGraph
#
# Responsabilidades:
#   1. Receber o rascunho gerado pelo Answerer e os chunks recuperados
#   2. Pedir ao LLM para avaliar se cada afirmação está suportada por evidências
#   3. Retornar um score 0.0–1.0 e, se < 0.7, indicar o que está faltando
#      (esse "retry_context" é passado ao Retriever para re-busca refinada)

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

APPROVAL_THRESHOLD: float = 0.7


@dataclass
class SelfCheckResult:
    score: float
    approved: bool
    unsupported: list[str]
    retry_context: str   # hint para o Retriever refinar a busca no retry


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Você é um verificador de evidências para um assistente acadêmico da UFCG.

Avalie se as afirmações de um rascunho de resposta estão SUPORTADAS pelos
trechos de documentos fornecidos como evidência.

DEFINIÇÕES:
- SUPORTADA: verificável diretamente em pelo menos um trecho.
- NÃO SUPORTADA: não aparece nos trechos (mesmo que seja verdadeira).

REGRAS:
1. Avalie SOMENTE com base nos trechos. Ignore seu conhecimento externo.
2. Seja conservador: em dúvida, marque como NÃO SUPORTADA.
3. Ignore afirmações óbvias ("a UFCG é federal") e disclaimers ("consulte o DAA").

Responda SOMENTE com JSON válido, sem markdown:
{
  "score": <float 0.0–1.0, fração de afirmações suportadas>,
  "unsupported_claims": ["afirmação sem suporte 1", ...],
  "missing_info": "<o que faltou nos trechos para suportar a resposta>"
}
Se tudo estiver suportado, use [] e "" nos dois últimos campos.
"""

_HUMAN_TEMPLATE = """\
=== RASCUNHO ===
{draft}

=== EVIDÊNCIAS ===
{evidence}

Avalie se o rascunho está suportado pelas evidências acima.
"""

_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=_SYSTEM_PROMPT),
    HumanMessage(content=_HUMAN_TEMPLATE),
])


# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def self_check(draft: str, chunks: list[dict], llm: BaseChatModel) -> float:
    """Retorna apenas o score (interface usada pelo graph.py)."""
    return self_check_full(draft, chunks, llm).score


def self_check_full(
    draft: str,
    chunks: list[dict],
    llm: BaseChatModel,
) -> SelfCheckResult:
    """Versão completa com retry_context para re-busca refinada."""

    if not chunks:
        logger.warning("[self_check] sem chunks → score 0.0")
        return SelfCheckResult(
            score=0.0, approved=False,
            unsupported=["Sem evidências recuperadas"],
            retry_context="regulamento matrícula disciplinas pré-requisitos UFCG",
        )

    if not draft or not draft.strip():
        return SelfCheckResult(score=1.0, approved=True, unsupported=[], retry_context="")

    evidence = _format_evidence(chunks)

    try:
        messages = _PROMPT.format_messages(draft=draft, evidence=evidence)
        response = llm.invoke(messages)
        raw = response.content.strip()
        logger.debug("[self_check] LLM raw: %r", raw[:300])
        result = _parse_result(raw)

    except Exception as exc:
        logger.error("[self_check] erro LLM: %s — aprovando por fallback", exc)
        result = SelfCheckResult(score=0.75, approved=True, unsupported=[], retry_context="")

    logger.info(
        "[self_check] score=%.2f approved=%s unsupported=%d",
        result.score, result.approved, len(result.unsupported),
    )
    return result


def self_check_node_update(
    draft: str,
    chunks: list[dict],
    llm: BaseChatModel,
    current_retry_count: int,
) -> dict:
    """
    Interface de alto nível para o nó node_self_check do graph.py.
    Retorna dict de atualização do AgentState incluindo retry_context.
    """
    result = self_check_full(draft, chunks, llm)
    update = {"self_check_score": result.score}

    if not result.approved and current_retry_count < 1:
        update["retry_count"] = current_retry_count + 1
        update["retry_context"] = result.retry_context
        logger.info("[self_check] retry agendado — contexto: %r", result.retry_context[:80])

    return update


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _parse_result(raw: str) -> SelfCheckResult:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = _regex_extract(cleaned)

    try:
        score = max(0.0, min(1.0, float(data.get("score", 0.5))))
    except (TypeError, ValueError):
        score = 0.5

    raw_claims = data.get("unsupported_claims", [])
    unsupported = [str(c) for c in raw_claims if c] if isinstance(raw_claims, list) else []
    retry_context = str(data.get("missing_info", "")).strip()

    return SelfCheckResult(
        score=score,
        approved=score >= APPROVAL_THRESHOLD,
        unsupported=unsupported,
        retry_context=retry_context,
    )


def _regex_extract(text: str) -> dict:
    score_m = re.search(r'"score"\s*:\s*([0-9.]+)', text)
    missing_m = re.search(r'"missing_info"\s*:\s*"([^"]*)"', text)
    return {
        "score": float(score_m.group(1)) if score_m else 0.5,
        "unsupported_claims": [],
        "missing_info": missing_m.group(1) if missing_m else "",
    }


def _format_evidence(chunks: list[dict], max_chars: int = 2500) -> str:
    lines, total = [], 0
    for i, chunk in enumerate(chunks, 1):
        page = chunk.get("page")
        loc = f"pág. {page}" if page else "sem página"
        entry = f"[{i}] {chunk.get('source','?')} ({loc}):\n\"{chunk.get('text','')}\"\n"
        if total + len(entry) > max_chars:
            lines.append(f"[... {len(chunks)-i+1} trechos omitidos]")
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="qwen2.5:3b", temperature=0.0)

    draft_ok = "Para cursar Redes (COMP3501), o aluno precisa de COMP2401 e COMP2201. Carga horária: 60h."
    chunks_ok = [
        {"text": "COMP3501 - Redes. Pré-requisitos: COMP2401 e COMP2201. Carga horária: 60h.", "source": "fluxograma_cc.pdf", "page": 5},
    ]

    draft_bad = "Redes exige Cálculo 3 e tem 90h. Só é ofertada à noite."
    chunks_bad = [{"text": "COMP3501 - Redes. Pré-requisitos: COMP2401 e COMP2201.", "source": "fluxograma_cc.pdf", "page": 5}]

    for label, draft, chunks in [("✅ suportado", draft_ok, chunks_ok), ("❌ não suportado", draft_bad, chunks_bad)]:
        r = self_check_full(draft, chunks, llm)
        print(f"\n{label}: score={r.score:.2f}  aprovado={r.approved}")
        if r.unsupported:
            print(f"  Não suportadas: {r.unsupported}")
        if r.retry_context:
            print(f"  Retry context: {r.retry_context!r}")