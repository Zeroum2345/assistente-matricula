from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mcp-docstore")

# Arquivo de log de auditoria (controle de acesso a tools)
AUDIT_LOG_PATH = Path(os.getenv("MCP_AUDIT_LOG", "logs/mcp_audit.jsonl"))
AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Allowlist explícita de tools permitidas (segurança)
ALLOWED_TOOLS: set[str] = {"search_docs", "get_prerequisites", "get_schedule"}

mcp = FastMCP(
    "ufcg-docstore",
    description=(
        "Acesso ao corpus de documentos públicos da UFCG: regulamentos, "
        "fluxogramas curriculares, pré-requisitos e horários de disciplinas."
    ),
)


# ---------------------------------------------------------------------------
# Middleware de auditoria
# ---------------------------------------------------------------------------

def _audit_log(tool: str, params: dict, result_summary: str) -> None:
    """Registra cada chamada de tool em JSONL para auditoria."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool": tool,
        "params": params,
        "result_summary": result_summary[:200],
    }
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("Falha ao registrar auditoria: %s", exc)


def _check_allowlist(tool: str) -> None:
    """Garante que a tool está na allowlist. Levanta ValueError se não estiver."""
    if tool not in ALLOWED_TOOLS:
        raise ValueError(
            f"Tool '{tool}' não está na allowlist. "
            f"Tools permitidas: {sorted(ALLOWED_TOOLS)}"
        )


# ---------------------------------------------------------------------------
# Tool 1 — Busca semântica geral
# ---------------------------------------------------------------------------

@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> list[dict]:
    """
    Realiza busca semântica no corpus de documentos públicos da UFCG.

    Args:
        query:  Texto da consulta em português.
        top_k:  Número máximo de resultados (1–10). Padrão: 5.

    Returns:
        Lista de trechos relevantes com metadados:
        [{text, source, page, score, section, excerpt}]

    Restrições de segurança:
        - Acesso somente leitura
        - top_k limitado a no máximo 10
        - Não retorna dados pessoais de alunos
    """
    _check_allowlist("search_docs")

    # Sanitiza parâmetros
    query = str(query).strip()[:500]   # limita tamanho da query
    top_k = max(1, min(int(top_k), 10))  # limita entre 1 e 10

    if not query:
        _audit_log("search_docs", {"query": "", "top_k": top_k}, "erro: query vazia")
        return []

    t0 = time.time()

    # Importa o retriever do projeto principal
    # (lazy import para não carregar o modelo na inicialização do servidor)
    from src.agents.retriever import retrieve_chunks
    chunks = retrieve_chunks(query=query, top_k=top_k)

    elapsed = time.time() - t0
    summary = f"{len(chunks)} chunks em {elapsed:.2f}s"
    _audit_log("search_docs", {"query": query[:100], "top_k": top_k}, summary)
    logger.info("[search_docs] query=%r  results=%d  (%.2fs)", query[:60], len(chunks), elapsed)

    return chunks


# ---------------------------------------------------------------------------
# Tool 2 — Pré-requisitos de uma disciplina
# ---------------------------------------------------------------------------

@mcp.tool()
def get_prerequisites(course_code: str) -> dict:
    """
    Retorna os pré-requisitos de uma disciplina a partir do fluxograma
    e regulamento indexados da UFCG.

    Args:
        course_code: Código da disciplina (ex: "COMP3501", "MAT2001").

    Returns:
        {
          "course_code": str,
          "prerequisites": [str],   # lista de códigos/nomes de pré-requisitos
          "sources": [dict],        # trechos dos documentos usados
          "found": bool             # False se não encontrou nos documentos
        }

    Restrições de segurança:
        - Acesso somente leitura
        - course_code sanitizado (apenas alfanumérico)
    """
    _check_allowlist("get_prerequisites")

    # Sanitiza: mantém apenas alfanumérico e traço/underline
    import re
    course_code = re.sub(r"[^A-Za-z0-9_\-]", "", str(course_code)).upper()[:20]

    if not course_code:
        _audit_log("get_prerequisites", {"course_code": ""}, "erro: código vazio")
        return {"course_code": "", "prerequisites": [], "sources": [], "found": False}

    from src.agents.retriever import retrieve_by_course_code
    from langchain_ollama import ChatOllama
    from src.agents.automation import _extract_prereqs_from_chunks

    chunks = retrieve_by_course_code(course_code, top_k=4)

    if not chunks:
        _audit_log("get_prerequisites", {"course_code": course_code}, "não encontrado")
        return {
            "course_code": course_code,
            "prerequisites": [],
            "sources": [],
            "found": False,
        }

    llm = ChatOllama(model="qwen2.5:3b", temperature=0.0)
    prereqs = _extract_prereqs_from_chunks(course_code, chunks, llm)

    result = {
        "course_code": course_code,
        "prerequisites": prereqs,
        "sources": [
            {"source": c.get("source"), "page": c.get("page"), "excerpt": c.get("excerpt", "")}
            for c in chunks[:3]
        ],
        "found": True,
    }

    _audit_log("get_prerequisites", {"course_code": course_code}, f"prereqs={prereqs}")
    logger.info("[get_prerequisites] %s → %s", course_code, prereqs)
    return result


# ---------------------------------------------------------------------------
# Tool 3 — Horários de uma disciplina
# ---------------------------------------------------------------------------

@mcp.tool()
def get_schedule(course_code: str, semester: str = "") -> dict:
    """
    Retorna os horários de uma disciplina para um semestre específico,
    com base nos documentos indexados.

    Args:
        course_code: Código da disciplina (ex: "COMP3501").
        semester:    Período letivo (ex: "2025.1"). Se vazio, retorna o mais recente.

    Returns:
        {
          "course_code": str,
          "semester": str,
          "schedule_text": str,    # ex: "seg/qua 08:00–10:00, sala LCC1"
          "days": [str],
          "times": [str],
          "room": str,
          "sources": [dict],
          "found": bool
        }

    Restrições de segurança:
        - Acesso somente leitura
        - Parâmetros sanitizados
    """
    _check_allowlist("get_schedule")

    import re
    course_code = re.sub(r"[^A-Za-z0-9_\-]", "", str(course_code)).upper()[:20]
    semester = re.sub(r"[^0-9.]", "", str(semester))[:10]

    if not course_code:
        return {"course_code": "", "semester": semester, "schedule_text": "código inválido",
                "days": [], "times": [], "room": "", "sources": [], "found": False}

    from src.agents.retriever import retrieve_by_course_code
    from langchain_ollama import ChatOllama
    from src.agents.automation import _extract_schedule_from_chunks

    # Inclui o semestre na query para afinidade semântica
    query_suffix = f" horário {semester}" if semester else " horário turma sala"
    from src.agents.retriever import retrieve_chunks
    chunks = retrieve_chunks(f"{course_code}{query_suffix}", top_k=4)

    if not chunks:
        _audit_log("get_schedule", {"course_code": course_code, "semester": semester}, "não encontrado")
        return {
            "course_code": course_code, "semester": semester,
            "schedule_text": "horário não encontrado nos documentos indexados",
            "days": [], "times": [], "room": "", "sources": [], "found": False,
        }

    llm = ChatOllama(model="qwen2.5:3b", temperature=0.0)
    schedule = _extract_schedule_from_chunks(course_code, chunks, llm)

    result = {
        "course_code": course_code,
        "semester": semester or "mais recente disponível",
        "schedule_text": schedule.get("schedule_text", "não encontrado"),
        "days": schedule.get("days", []),
        "times": schedule.get("times", []),
        "room": schedule.get("room", ""),
        "sources": [
            {"source": c.get("source"), "page": c.get("page"), "excerpt": c.get("excerpt", "")}
            for c in chunks[:2]
        ],
        "found": schedule.get("schedule_text", "") not in ("", "não encontrado"),
    }

    _audit_log("get_schedule", {"course_code": course_code, "semester": semester},
               f"schedule={result['schedule_text']}")
    logger.info("[get_schedule] %s %s → %s", course_code, semester, result["schedule_text"])
    return result


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logger.info("Iniciando MCP server ufcg-docstore...")
    logger.info("Tools disponíveis: %s", sorted(ALLOWED_TOOLS))
    logger.info("Audit log: %s", AUDIT_LOG_PATH.resolve())

    # Modo stdio (padrão MCP para integração com LangChain)
    mcp.run(transport="stdio")