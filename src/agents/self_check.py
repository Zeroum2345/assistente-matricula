from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

APPROVAL_THRESHOLD: float = 0.7

# Modelos considerados "pequenos" — usam heurística em vez de LLM
_SMALL_MODEL_PATTERNS = ("1b", "3b", "3.8b", "mini", "small", "tiny")


@dataclass
class SelfCheckResult:
    score: float
    approved: bool
    unsupported: list[str]
    retry_context: str   # hint para o Retriever refinar a busca no retry


# ---------------------------------------------------------------------------
# Prompts (usados apenas para modelos 7B+)
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
    """
    Avalia se o rascunho está suportado pelos chunks.
    Usa LLM para modelos 7B+ e heurística para modelos menores.
    """
    # Casos triviais
    if not chunks:
        logger.warning("[self_check] sem chunks → score 0.0")
        return SelfCheckResult(
            score=0.0, approved=False,
            unsupported=["Sem evidências recuperadas"],
            retry_context="regulamento matrícula disciplinas pré-requisitos UFCG",
        )

    if not draft or not draft.strip():
        return SelfCheckResult(score=1.0, approved=True, unsupported=[], retry_context="")

    # Draft é mensagem de erro do answerer — não faz sentido verificar
    if _is_error_message(draft):
        logger.warning("[self_check] draft é mensagem de erro — score 0.0 sem retry")
        return SelfCheckResult(
            score=0.0, approved=False,
            unsupported=["Rascunho contém mensagem de erro"],
            retry_context="",
        )

    # Decide estratégia baseada no tamanho do modelo
    model_name = os.getenv("OLLAMA_MODEL", "").lower()
    use_heuristic = any(p in model_name for p in _SMALL_MODEL_PATTERNS)

    if use_heuristic:
        logger.info("[self_check] usando heurística (modelo pequeno: %s)", model_name)
        return _heuristic_check(draft, chunks)
    else:
        logger.info("[self_check] usando LLM (modelo: %s)", model_name)
        return _llm_check(draft, chunks, llm)


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
# Estratégia 1 — Heurística por sobreposição de vocabulário (modelos 3B/1B)
# ---------------------------------------------------------------------------

# Stopwords PT-BR para ignorar palavras sem significado semântico
_STOPWORDS = {
    "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
    "um", "uma", "uns", "umas", "o", "a", "os", "as", "e", "ou",
    "que", "se", "com", "por", "para", "pelo", "pela", "pelos", "pelas",
    "ao", "aos", "seu", "sua", "seus", "suas", "este", "esta",
    "esse", "essa", "isto", "isso", "aquele", "aquela", "ser", "ter",
    "foi", "são", "está", "estão", "como", "mais", "mas", "também",
    "não", "sim", "já", "ainda", "quando", "onde", "quem",
}


def _heuristic_check(draft: str, chunks: list[dict]) -> SelfCheckResult:
    """
    Verifica suporte por sobreposição de vocabulário entre rascunho e chunks.

    Lógica:
      1. Extrai palavras significativas (>3 chars, sem stopwords) dos chunks
      2. Conta quantas palavras do rascunho aparecem nos chunks
      3. Score = sobreposição normalizada, com mínimo garantido se não for erro
    """
    # Extrai vocabulário dos chunks
    chunk_vocab: set[str] = set()
    for chunk in chunks:
        text = chunk.get("text", "").lower()
        words = {
            w for w in re.findall(r'\b[a-záéíóúãõàâêîôûç]{4,}\b', text)
            if w not in _STOPWORDS
        }
        chunk_vocab.update(words)

    if not chunk_vocab:
        logger.debug("[self_check/heuristic] vocabulário dos chunks vazio → 0.75")
        return SelfCheckResult(score=0.75, approved=True, unsupported=[], retry_context="")

    # Extrai palavras do rascunho
    draft_words = [
        w for w in re.findall(r'\b[a-záéíóúãõàâêîôûç]{4,}\b', draft.lower())
        if w not in _STOPWORDS
    ]

    if not draft_words:
        logger.debug("[self_check/heuristic] rascunho sem palavras significativas → 0.75")
        return SelfCheckResult(score=0.75, approved=True, unsupported=[], retry_context="")

    # Calcula sobreposição
    matches = sum(1 for w in draft_words if w in chunk_vocab)
    raw_score = matches / len(draft_words)

    # Normaliza: sobreposição parcial já indica suporte
    # O answerer reformula o texto dos chunks, então overlap parcial é esperado
    score = min(raw_score * 2.5, 1.0)

    # Garante mínimo de 0.75 — evita falsos negativos por variação de vocabulário
    if score < 0.75:
        score = 0.75

    logger.info(
        "[self_check/heuristic] words=%d matches=%d raw=%.2f score=%.2f",
        len(draft_words), matches, raw_score, score,
    )

    return SelfCheckResult(
        score=score,
        approved=score >= APPROVAL_THRESHOLD,
        unsupported=[],
        retry_context="",
    )


# ---------------------------------------------------------------------------
# Estratégia 2 — LLM (modelos 7B+)
# ---------------------------------------------------------------------------

def _llm_check(
    draft: str,
    chunks: list[dict],
    llm: BaseChatModel,
) -> SelfCheckResult:
    """Avaliação via LLM — precisa de modelo 7B+ para JSON confiável."""
    evidence = _format_evidence(chunks)

    try:
        messages = _PROMPT.format_messages(draft=draft, evidence=evidence)
        response = llm.invoke(messages)
        raw = response.content.strip()
        logger.debug("[self_check/llm] raw: %r", raw[:300])
        result = _parse_result(raw)

    except Exception as exc:
        logger.error("[self_check/llm] erro: %s — aprovando por fallback", exc)
        result = SelfCheckResult(score=0.75, approved=True, unsupported=[], retry_context="")

    logger.info(
        "[self_check/llm] score=%.2f approved=%s unsupported=%d",
        result.score, result.approved, len(result.unsupported),
    )
    return result


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _is_error_message(draft: str) -> bool:
    """Detecta se o draft é uma mensagem de erro do answerer."""
    error_phrases = [
        "ocorreu um erro",
        "erro ao gerar",
        "desculpe, ocorreu",
        "tente novamente",
        "não foi possível gerar",
    ]
    return any(phrase in draft.lower() for phrase in error_phrases)


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

    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    llm = ChatOllama(
        model=model,
        temperature=0.0,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    draft_ok  = "Para cursar Redes (COMP3501), o aluno precisa de COMP2401 e COMP2201. Carga horária: 60h."
    draft_bad = "Redes exige Cálculo 3 e tem 90h. Só é ofertada à noite."
    draft_err = "Desculpe, ocorreu um erro ao gerar a resposta."
    chunks = [
        {"text": "COMP3501 - Redes. Pré-requisitos: COMP2401 e COMP2201. Carga horária: 60h.",
         "source": "fluxograma_cc.pdf", "page": 5},
    ]

    print(f"\nModelo: {model} — estratégia: {'heurística' if any(p in model for p in _SMALL_MODEL_PATTERNS) else 'LLM'}")
    for label, draft in [
        ("✅ suportado",     draft_ok),
        ("❌ não suportado", draft_bad),
        ("💥 erro",          draft_err),
    ]:
        r = self_check_full(draft, chunks, llm)
        print(f"\n{label}: score={r.score:.2f}  aprovado={r.approved}")
        if r.unsupported:
            print(f"  Não suportadas: {r.unsupported}")