# Substitui automation.py quando a Eureca estiver configurada.
# Se EURECA_LOGIN não estiver definido, cai automaticamente para o FAISS.

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel

from src.agents.automation import (
    _error_result,
    _deduplicate_sources,
    _extract_course_codes,
    _extract_target_course,
    _run_trail,           # trilha continua usando FAISS (sem equivalente direto na API)
)
from src.integrations.eureca_client import EurecaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instância do cliente Eureca (lazy — só autentica quando necessário)
# ---------------------------------------------------------------------------

_eureca_client: EurecaClient | None = None


def _get_eureca() -> EurecaClient | None:
    """Retorna cliente Eureca ou None se não configurado."""
    global _eureca_client
    try:
        if _eureca_client is None:
            _eureca_client = EurecaClient()
        # Valida que as credenciais estão configuradas
        _ = _eureca_client.token
        return _eureca_client
    except ValueError:
        logger.info("[automation_eureca] Eureca não configurada — usando FAISS")
        return None
    except Exception as exc:
        logger.warning("[automation_eureca] Eureca indisponível: %s — usando FAISS", exc)
        return None


# ---------------------------------------------------------------------------
# Dispatcher principal (substitui run_automation do automation.py)
# ---------------------------------------------------------------------------

def run_automation_with_eureca(
    automation_type: str | None,
    query: str,
    context_chunks: list[dict] | None = None,
    periodo: str = "2025.1",
) -> dict:
    """
    Executa automação com fallback Eureca → FAISS.

    Args:
        automation_type: "prereq", "schedule" ou "trail"
        query:           Query original do usuário
        context_chunks:  Chunks do FAISS (usados como fallback e para trilha)
        periodo:         Período letivo para consulta de horários (ex: "2025.1")
    """
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="qwen2.5:3b", temperature=0.0)
    chunks = context_chunks or []
    eureca = _get_eureca()

    if automation_type == "prereq":
        return _run_prereq_eureca(query, chunks, llm, eureca)
    elif automation_type == "schedule":
        return _run_schedule_eureca(query, chunks, llm, eureca, periodo)
    elif automation_type == "trail":
        return _run_trail(query, chunks, llm)   # trilha usa FAISS
    else:
        from src.agents.automation import run_automation
        return run_automation(automation_type, query, chunks)


# ---------------------------------------------------------------------------
# Verificação de pré-requisitos com Eureca
# ---------------------------------------------------------------------------

def _run_prereq_eureca(
    query: str,
    chunks: list[dict],
    llm: BaseChatModel,
    eureca: EurecaClient | None,
) -> dict:
    """
    Verifica pré-requisitos com dados da API Eureca.
    Fallback automático para FAISS se Eureca indisponível.
    """
    from src.agents.automation import _run_prereq, _extract_prereq_params

    if eureca is None:
        logger.info("[automation_eureca/prereq] fallback → FAISS")
        return _run_prereq(query, chunks, llm)

    # Extrai parâmetros da query
    params            = _extract_prereq_params(query, llm)
    target_courses    = params.get("target_courses", [])
    completed_courses = params.get("completed_courses", [])

    if not target_courses:
        return _error_result(
            "Não consegui identificar as disciplinas. "
            "Exemplo: 'Posso cursar COMP3501 se já fiz COMP2401?'"
        )

    results     = []
    all_sources = []

    for course in target_courses:
        # Tenta Eureca primeiro
        prereqs_ok = False
        prereqs    = []
        source_tag = "faiss"

        try:
            prereqs    = eureca.get_prerequisitos(course)
            prereqs_ok = True
            source_tag = "eureca_api"
            logger.info("[automation_eureca/prereq] %s: prereqs=%s (Eureca)", course, prereqs)
        except Exception as exc:
            logger.warning("[automation_eureca/prereq] Eureca falhou para %s: %s — usando FAISS", course, exc)

        # Fallback para FAISS
        if not prereqs_ok:
            from src.agents.retriever import retrieve_by_course_code
            from src.agents.automation import _extract_prereqs_from_chunks
            course_chunks = retrieve_by_course_code(course, top_k=4)
            all_sources.extend(course_chunks)
            prereqs = _extract_prereqs_from_chunks(course, course_chunks, llm)
            source_tag = "faiss"

        missing   = [p for p in prereqs if p not in completed_courses]
        satisfied = [p for p in prereqs if p in completed_courses]

        status = "✅ pode cursar" if not missing else "❌ pré-requisitos faltando"
        results.append({
            "course":                   course,
            "status":                   status,
            "prerequisites_required":   prereqs,
            "prerequisites_satisfied":  satisfied,
            "prerequisites_missing":    missing,
            "fonte":                    source_tag,
        })

    can_all = all(not r["prerequisites_missing"] for r in results)
    summary = (
        "Você pode se matricular em todas as disciplinas solicitadas. ✅"
        if can_all
        else "Existem pré-requisitos não satisfeitos para algumas disciplinas. ❌"
    )
    details = [
        f"**{r['course']}** [{r['fonte']}]: {r['status']}"
        + (f" — faltam: {', '.join(r['prerequisites_missing'])}" if r["prerequisites_missing"] else "")
        for r in results
    ]

    return {
        "type":    "prereq",
        "summary": summary,
        "details": details,
        "raw":     results,
        "sources": _deduplicate_sources(all_sources),
        "blocked": False,
        "block_reason": "",
    }


# ---------------------------------------------------------------------------
# Detecção de conflito com Eureca
# ---------------------------------------------------------------------------

def _run_schedule_eureca(
    query: str,
    chunks: list[dict],
    llm: BaseChatModel,
    eureca: EurecaClient | None,
    periodo: str,
) -> dict:
    """
    Detecta conflitos de horário com dados reais da API Eureca.
    Fallback automático para FAISS se Eureca indisponível.
    """
    from src.agents.automation import _run_schedule

    if eureca is None:
        logger.info("[automation_eureca/schedule] fallback → FAISS")
        return _run_schedule(query, chunks, llm)

    # Extrai período da query se mencionado (ex: "2025.1", "2025/1")
    import re
    periodo_match = re.search(r"20\d\d[./][12]", query)
    if periodo_match:
        periodo = periodo_match.group(0).replace("/", ".")

    # Extrai códigos de disciplinas
    course_codes = _extract_course_codes(query, llm)
    if len(course_codes) < 2:
        return _error_result(
            "Preciso de pelo menos 2 códigos de disciplinas para verificar conflitos. "
            f"Exemplo: 'COMP3501, MAT2001 e COMP2401 têm conflito em {periodo}?'"
        )

    try:
        conflitos = eureca.verificar_conflito(course_codes, periodo)

        if conflitos:
            summary = f"⚠️ {len(conflitos)} conflito(s) detectado(s) no período {periodo}."
            details = [c["descricao"] for c in conflitos]
        else:
            summary = f"✅ Nenhum conflito de horário no período {periodo}."
            # Monta detalhes com os horários de cada disciplina
            details = []
            for code in course_codes:
                horarios = eureca.get_horarios(code, periodo)
                if horarios:
                    slots = " | ".join(
                        f"{h['dia']} {h['hora_inicio']}–{h['hora_fim']}"
                        for h in horarios[:3]
                    )
                    details.append(f"**{code}**: {slots}")
                else:
                    details.append(f"**{code}**: horário não disponível na API")

        return {
            "type":    "schedule",
            "summary": summary,
            "details": details,
            "raw":     {"conflitos": conflitos, "periodo": periodo, "codigos": course_codes},
            "sources": [{"source": "eureca_api", "page": None,
                         "excerpt": f"Dados do período {periodo}"}],
            "blocked": False,
            "block_reason": "",
        }

    except Exception as exc:
        logger.warning("[automation_eureca/schedule] Eureca falhou: %s — usando FAISS", exc)
        return _run_schedule(query, chunks, llm)
