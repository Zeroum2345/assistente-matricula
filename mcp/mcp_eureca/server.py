from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.integrations.eureca_client import EurecaClient, _cache

logger = logging.getLogger("mcp-eureca")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

AUDIT_LOG = Path(os.getenv("MCP_EURECA_AUDIT_LOG", "logs/mcp_eureca_audit.jsonl"))
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

ALLOWED_TOOLS = {
    "get_prerequisitos_eureca",
    "get_horarios_eureca",
    "verificar_conflito_eureca",
    "get_turmas_eureca",
}


def _audit(tool: str, params: dict, summary: str) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool":      tool,
        "params":    params,
        "summary":   summary[:200],
    }
    try:
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("Falha no audit log: %s", exc)


def _check_allow(tool: str) -> None:
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool '{tool}' não está na allowlist.")


def _sanitize_code(code: str) -> str:
    """Remove caracteres não-alfanuméricos de código de disciplina."""
    return re.sub(r"[^A-Za-z0-9_\-]", "", str(code)).upper()[:20]


def _sanitize_periodo(periodo: str) -> str:
    """Valida formato YYYY.N (ex: 2025.1)."""
    periodo = re.sub(r"[^0-9.]", "", str(periodo))[:7]
    if not re.fullmatch(r"\d{4}\.[12]", periodo):
        raise ValueError(f"Período inválido: {periodo!r}. Use formato YYYY.N (ex: 2025.1)")
    return periodo


def _sanitize_curriculo(curriculo_id: str) -> str:
    return re.sub(r"[^0-9]", "", str(curriculo_id))[:12]


# ---------------------------------------------------------------------------
# Cliente Eureca singleton (usa credenciais do .env)
# ---------------------------------------------------------------------------

_client: EurecaClient | None = None


def _get_client() -> EurecaClient:
    global _client
    if _client is None:
        _client = EurecaClient()
    return _client


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "ufcg-eureca",
    description=(
        "Acesso em tempo real à API Eureca da UFCG: pré-requisitos, "
        "horários, turmas e conflitos de matrícula com dados oficiais."
    ),
)


@mcp.tool()
def get_prerequisitos_eureca(
    codigo: str,
    curriculo_id: str = "14102100",
) -> dict:
    """
    Retorna os pré-requisitos oficiais de uma disciplina diretamente da API Eureca.
    Dados em tempo real — mais confiáveis que o índice FAISS.

    Args:
        codigo:       Código da disciplina (ex: "COMP3501")
        curriculo_id: Código do currículo (padrão: "14102100" = CC/UFCG)

    Returns:
        {
          "codigo": str,
          "curriculo_id": str,
          "prerequisitos": [str],
          "fonte": "eureca_api",
          "found": bool
        }

    Segurança: somente leitura, sem dados pessoais.
    """
    _check_allow("get_prerequisitos_eureca")
    codigo       = _sanitize_code(codigo)
    curriculo_id = _sanitize_curriculo(curriculo_id)

    if not codigo:
        return {"error": "código de disciplina inválido"}

    t0 = time.time()
    try:
        client  = _get_client()
        prereqs = client.get_prerequisitos(codigo, curriculo_id)
        elapsed = round(time.time() - t0, 2)

        result = {
            "codigo":       codigo,
            "curriculo_id": curriculo_id,
            "prerequisitos": prereqs,
            "fonte":        "eureca_api",
            "found":        prereqs is not None,
            "elapsed_s":    elapsed,
        }
        _audit("get_prerequisitos_eureca",
               {"codigo": codigo, "curriculo_id": curriculo_id},
               f"prereqs={prereqs} ({elapsed}s)")
        logger.info("[mcp-eureca] prereqs %s → %s", codigo, prereqs)
        return result

    except Exception as exc:
        _audit("get_prerequisitos_eureca",
               {"codigo": codigo}, f"erro: {exc}")
        logger.error("[mcp-eureca] erro prereqs %s: %s", codigo, exc)
        return {
            "codigo":        codigo,
            "prerequisitos": [],
            "found":         False,
            "error":         str(exc),
            "fonte":         "eureca_api",
        }


@mcp.tool()
def get_horarios_eureca(
    codigo: str,
    periodo: str,
) -> dict:
    """
    Retorna os horários reais de uma disciplina em um período letivo,
    com sala, professor e vagas disponíveis.

    Args:
        codigo:  Código da disciplina (ex: "COMP3501")
        periodo: Período letivo (ex: "2025.1")

    Returns:
        {
          "codigo": str,
          "periodo": str,
          "horarios": [{turma, dia, hora_inicio, hora_fim, sala, professor, vagas}],
          "total_turmas": int,
          "fonte": "eureca_api"
        }
    """
    _check_allow("get_horarios_eureca")
    codigo  = _sanitize_code(codigo)
    periodo = _sanitize_periodo(periodo)

    t0 = time.time()
    try:
        client   = _get_client()
        horarios = client.get_horarios(codigo, periodo)
        elapsed  = round(time.time() - t0, 2)

        result = {
            "codigo":       codigo,
            "periodo":      periodo,
            "horarios":     horarios,
            "total_turmas": len({h["turma"] for h in horarios}),
            "fonte":        "eureca_api",
            "elapsed_s":    elapsed,
        }
        _audit("get_horarios_eureca",
               {"codigo": codigo, "periodo": periodo},
               f"{len(horarios)} horários ({elapsed}s)")
        logger.info("[mcp-eureca] horarios %s/%s → %d slots", codigo, periodo, len(horarios))
        return result

    except Exception as exc:
        _audit("get_horarios_eureca",
               {"codigo": codigo, "periodo": periodo}, f"erro: {exc}")
        logger.error("[mcp-eureca] erro horarios %s: %s", codigo, exc)
        return {
            "codigo":   codigo,
            "periodo":  periodo,
            "horarios": [],
            "error":    str(exc),
            "fonte":    "eureca_api",
        }


@mcp.tool()
def verificar_conflito_eureca(
    codigos: list[str],
    periodo: str,
) -> dict:
    """
    Verifica conflitos de horário entre disciplinas usando dados reais da API Eureca.

    Args:
        codigos: Lista de códigos de disciplinas (ex: ["COMP3501", "MAT2001"])
        periodo: Período letivo (ex: "2025.1")

    Returns:
        {
          "periodo": str,
          "codigos_verificados": [str],
          "conflitos": [{curso_a, curso_b, dia, hora_a, hora_b, descricao}],
          "tem_conflito": bool,
          "resumo": str,
          "fonte": "eureca_api"
        }
    """
    _check_allow("verificar_conflito_eureca")
    codigos_clean = [_sanitize_code(c) for c in codigos if c][:10]  # max 10
    periodo       = _sanitize_periodo(periodo)

    if len(codigos_clean) < 2:
        return {"error": "Informe pelo menos 2 códigos de disciplinas."}

    t0 = time.time()
    try:
        client    = _get_client()
        conflitos = client.verificar_conflito(codigos_clean, periodo)
        elapsed   = round(time.time() - t0, 2)

        tem_conflito = len(conflitos) > 0
        resumo = (
            f"⚠️ {len(conflitos)} conflito(s) detectado(s) entre as disciplinas."
            if tem_conflito
            else "✅ Nenhum conflito de horário detectado."
        )

        result = {
            "periodo":             periodo,
            "codigos_verificados": codigos_clean,
            "conflitos":           conflitos,
            "tem_conflito":        tem_conflito,
            "resumo":              resumo,
            "fonte":               "eureca_api",
            "elapsed_s":           elapsed,
        }
        _audit("verificar_conflito_eureca",
               {"codigos": codigos_clean, "periodo": periodo},
               f"conflitos={len(conflitos)} ({elapsed}s)")
        logger.info("[mcp-eureca] conflito %s: %d conflitos", codigos_clean, len(conflitos))
        return result

    except Exception as exc:
        _audit("verificar_conflito_eureca",
               {"codigos": codigos_clean, "periodo": periodo}, f"erro: {exc}")
        return {
            "conflitos":    [],
            "tem_conflito": False,
            "error":        str(exc),
            "fonte":        "eureca_api",
        }


@mcp.tool()
def get_turmas_eureca(
    codigo: str,
    periodo: str,
) -> dict:
    """
    Lista todas as turmas disponíveis de uma disciplina num período,
    com vagas, professor e horários completos.

    Args:
        codigo:  Código da disciplina (ex: "COMP3501")
        periodo: Período letivo (ex: "2025.1")

    Returns:
        {
          "codigo": str,
          "periodo": str,
          "turmas": [{id, professor, vagas, horarios: [{dia, hora_inicio, hora_fim, sala}]}],
          "total": int,
          "fonte": "eureca_api"
        }
    """
    _check_allow("get_turmas_eureca")
    codigo  = _sanitize_code(codigo)
    periodo = _sanitize_periodo(periodo)

    t0 = time.time()
    try:
        client  = _get_client()
        turmas  = get_turmas_raw(codigo, periodo, client)
        elapsed = round(time.time() - t0, 2)

        result = {
            "codigo":   codigo,
            "periodo":  periodo,
            "turmas":   turmas,
            "total":    len(turmas),
            "fonte":    "eureca_api",
            "elapsed_s": elapsed,
        }
        _audit("get_turmas_eureca",
               {"codigo": codigo, "periodo": periodo},
               f"{len(turmas)} turmas ({elapsed}s)")
        return result

    except Exception as exc:
        _audit("get_turmas_eureca",
               {"codigo": codigo, "periodo": periodo}, f"erro: {exc}")
        return {"codigo": codigo, "periodo": periodo, "turmas": [], "error": str(exc)}


def get_turmas_raw(codigo: str, periodo: str, client: EurecaClient) -> list[dict]:
    """Retorna turmas estruturadas com horários agrupados por turma."""
    from src.integrations.eureca_client import get_turmas
    turmas_api = get_turmas(periodo=periodo, token=client.token, componente=codigo)

    result = []
    for t in turmas_api:
        professor = t.get("professor", {})
        prof_nome = professor.get("nome", "") if isinstance(professor, dict) else str(professor)

        from src.integrations.eureca_client import _parse_hora, _normalize_dia
        horarios = [
            {
                "dia":         _normalize_dia(h.get("dia", "")),
                "hora_inicio": _parse_hora(h.get("hora", ""))[0],
                "hora_fim":    _parse_hora(h.get("hora", ""))[1],
                "sala":        h.get("sala", "") or h.get("local", ""),
            }
            for h in t.get("horarios", [])
        ]

        result.append({
            "id":        t.get("turma") or t.get("id", ""),
            "professor": prof_nome,
            "vagas":     t.get("vagas", 0),
            "horarios":  horarios,
        })

    return result


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Iniciando MCP server ufcg-eureca...")
    logger.info("Tools: %s", sorted(ALLOWED_TOOLS))
    logger.info("Audit log: %s", AUDIT_LOG.resolve())
    mcp.run(transport="stdio")