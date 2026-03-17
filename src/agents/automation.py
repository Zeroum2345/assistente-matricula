from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.agents.retriever import retrieve_chunks, retrieve_by_course_code

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM interno (temperatura baixa para extração estruturada)
# ---------------------------------------------------------------------------

def _get_llm() -> BaseChatModel:
    return ChatOllama(model="qwen2.5:3b", temperature=0.0, num_ctx=8192)


# ---------------------------------------------------------------------------
# Dispatcher principal
# ---------------------------------------------------------------------------

def run_automation(
    automation_type: str | None,
    query: str,
    context_chunks: list[dict] | None = None,
) -> dict:
    """
    Executa a automação solicitada.

    Args:
        automation_type: "prereq", "schedule" ou "trail"
        query:           Query original do usuário (usada para extração de parâmetros)
        context_chunks:  Chunks já recuperados pelo Retriever (reutilizados)

    Returns:
        Dict com: type, summary, details, sources, blocked (bool), block_reason
    """
    llm = _get_llm()
    chunks = context_chunks or []

    dispatch = {
        "prereq":   _run_prereq,
        "schedule": _run_schedule,
        "trail":    _run_trail,
    }

    handler = dispatch.get(automation_type or "")
    if not handler:
        logger.warning("[automation] tipo desconhecido: %r", automation_type)
        return _error_result("Tipo de automação não reconhecido.")

    try:
        return handler(query, chunks, llm)
    except Exception as exc:
        logger.error("[automation] erro em %r: %s", automation_type, exc)
        return _error_result(f"Erro ao executar a automação: {exc}")


# ---------------------------------------------------------------------------
# Fluxo 1 — Verificação de pré-requisitos
# ---------------------------------------------------------------------------

def _run_prereq(query: str, chunks: list[dict], llm: BaseChatModel) -> dict:
    """
    Extrai da query:
      - disciplinas que o aluno QUER cursar
      - disciplinas que o aluno JÁ CURSOU
    Verifica no corpus se os pré-requisitos estão satisfeitos.
    """
    logger.info("[automation/prereq] iniciando")

    # 1. Extrai parâmetros da query
    params = _extract_prereq_params(query, llm)
    target_courses = params.get("target_courses", [])
    completed_courses = params.get("completed_courses", [])

    if not target_courses:
        return _error_result(
            "Não consegui identificar quais disciplinas você quer cursar. "
            "Exemplo: 'Posso cursar COMP3501 se já fiz COMP2401 e COMP2201?'"
        )

    # 2. Para cada disciplina-alvo, busca pré-requisitos no corpus
    results = []
    all_sources = []

    for course in target_courses:
        course_chunks = retrieve_by_course_code(course, top_k=4)
        all_sources.extend(course_chunks)

        prereqs = _extract_prereqs_from_chunks(course, course_chunks, llm)
        missing = [p for p in prereqs if p not in completed_courses]
        satisfied = [p for p in prereqs if p in completed_courses]

        status = "✅ pode cursar" if not missing else "❌ pré-requisitos faltando"
        detail = {
            "course": course,
            "status": status,
            "prerequisites_required": prereqs,
            "prerequisites_satisfied": satisfied,
            "prerequisites_missing": missing,
        }
        results.append(detail)

    # 3. Monta resumo
    can_all = all(not r["prerequisites_missing"] for r in results)
    summary_lines = [
        f"**{r['course']}**: {r['status']}"
        + (f" — faltam: {', '.join(r['prerequisites_missing'])}" if r["prerequisites_missing"] else "")
        for r in results
    ]
    summary = (
        "Você pode se matricular em todas as disciplinas solicitadas. ✅"
        if can_all
        else "Existem pré-requisitos não satisfeitos para algumas disciplinas. ❌"
    )

    return {
        "type": "prereq",
        "summary": summary,
        "details": summary_lines,
        "raw": results,
        "sources": _deduplicate_sources(all_sources),
        "blocked": False,
        "block_reason": "",
    }


def _extract_prereq_params(query: str, llm: BaseChatModel) -> dict:
    """Usa o LLM para extrair disciplinas-alvo e já cursadas da query."""
    prompt = f"""\
Da mensagem abaixo, extraia:
- "target_courses": lista de códigos/nomes de disciplinas que o aluno QUER cursar
- "completed_courses": lista de códigos/nomes de disciplinas que o aluno JÁ FEZ

Mensagem: "{query}"

Responda SOMENTE com JSON válido:
{{"target_courses": [...], "completed_courses": [...]}}
Se não conseguir identificar, use listas vazias.
"""
    try:
        response = llm.invoke(prompt)
        raw = re.sub(r"```(?:json)?", "", response.content).strip().strip("`")
        return json.loads(raw)
    except Exception as exc:
        logger.warning("[automation/prereq] extração de parâmetros falhou: %s", exc)
        # Fallback: tenta extrair códigos de disciplina via regex
        codes = re.findall(r"\b[A-Z]{2,6}\d{3,5}\b", query)
        return {"target_courses": codes[:3], "completed_courses": codes[3:]}


def _extract_prereqs_from_chunks(
    course: str, chunks: list[dict], llm: BaseChatModel
) -> list[str]:
    """Extrai a lista de pré-requisitos de uma disciplina a partir dos chunks."""
    if not chunks:
        return []

    evidence = "\n".join(
        f"[{i+1}] {c.get('text','')}" for i, c in enumerate(chunks[:3])
    )
    prompt = f"""\
Com base nos trechos abaixo, liste os pré-requisitos da disciplina "{course}".
Retorne SOMENTE uma lista JSON de strings com os códigos/nomes, ex: ["COMP2401", "COMP2201"].
Se não houver pré-requisitos ou não encontrar a informação, retorne [].

Trechos:
{evidence}
"""
    try:
        response = llm.invoke(prompt)
        raw = re.sub(r"```(?:json)?", "", response.content).strip().strip("`")
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Fluxo 2 — Detecção de conflito de horário
# ---------------------------------------------------------------------------

def _run_schedule(query: str, chunks: list[dict], llm: BaseChatModel) -> dict:
    """
    Extrai os códigos de disciplina da query, busca os horários no corpus
    e identifica sobreposições.
    """
    logger.info("[automation/schedule] iniciando")

    # 1. Extrai códigos de disciplina
    course_codes = _extract_course_codes(query, llm)
    if len(course_codes) < 2:
        return _error_result(
            "Preciso de pelo menos 2 códigos de disciplina para verificar conflitos. "
            "Exemplo: 'COMP3501, MAT2001 e COMP2401 têm conflito de horário?'"
        )

    # 2. Busca horários de cada disciplina no corpus
    schedule_data = {}
    all_sources = []

    for code in course_codes:
        course_chunks = retrieve_by_course_code(code, top_k=3)
        all_sources.extend(course_chunks)

        schedule = _extract_schedule_from_chunks(code, course_chunks, llm)
        schedule_data[code] = schedule

    # 3. Detecta conflitos
    conflicts = _detect_conflicts(schedule_data)

    if conflicts:
        summary = f"⚠️ Foram detectados {len(conflicts)} conflito(s) de horário."
        details = [
            f"**{c['course_a']}** × **{c['course_b']}**: {c['overlap']}"
            for c in conflicts
        ]
    else:
        summary = "✅ Nenhum conflito de horário detectado entre as disciplinas verificadas."
        details = [
            f"**{code}**: {sched.get('schedule_text', 'horário não encontrado nos documentos')}"
            for code, sched in schedule_data.items()
        ]

    return {
        "type": "schedule",
        "summary": summary,
        "details": details,
        "raw": {"schedules": schedule_data, "conflicts": conflicts},
        "sources": _deduplicate_sources(all_sources),
        "blocked": False,
        "block_reason": "",
    }


def _extract_course_codes(query: str, llm: BaseChatModel) -> list[str]:
    """Extrai códigos de disciplina da query via regex + LLM fallback."""
    # Tenta regex primeiro (mais rápido)
    codes = re.findall(r"\b[A-Z]{2,6}\d{3,5}\b", query)
    if len(codes) >= 2:
        return list(dict.fromkeys(codes))  # deduplica mantendo ordem

    # LLM fallback para queries em linguagem natural
    prompt = f"""\
Extraia os códigos ou nomes de disciplinas da mensagem abaixo.
Retorne SOMENTE uma lista JSON de strings: ["COMP3501", "MAT2001", ...]
Se não encontrar, retorne [].

Mensagem: "{query}"
"""
    try:
        response = llm.invoke(prompt)
        raw = re.sub(r"```(?:json)?", "", response.content).strip().strip("`")
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception:
        return codes


def _extract_schedule_from_chunks(
    course: str, chunks: list[dict], llm: BaseChatModel
) -> dict:
    """Extrai horário estruturado de uma disciplina a partir dos chunks."""
    if not chunks:
        return {"course": course, "days": [], "times": [], "schedule_text": "não encontrado"}

    evidence = "\n".join(f"[{i+1}] {c.get('text','')}" for i, c in enumerate(chunks[:3]))
    prompt = f"""\
Com base nos trechos abaixo, extraia o horário da disciplina "{course}".
Retorne SOMENTE JSON:
{{
  "course": "{course}",
  "days": ["seg", "qua"],
  "times": ["08:00-10:00", "08:00-10:00"],
  "schedule_text": "seg/qua 08:00–10:00"
}}
Se não encontrar o horário, use days=[], times=[] e schedule_text="não encontrado".

Trechos:
{evidence}
"""
    try:
        response = llm.invoke(prompt)
        raw = re.sub(r"```(?:json)?", "", response.content).strip().strip("`")
        return json.loads(raw)
    except Exception:
        return {"course": course, "days": [], "times": [], "schedule_text": "não encontrado"}


def _detect_conflicts(schedule_data: dict[str, dict]) -> list[dict]:
    """
    Detecta sobreposições de horário entre pares de disciplinas.
    Compara dias e faixas de horário.
    """
    courses = list(schedule_data.items())
    conflicts = []

    for i in range(len(courses)):
        for j in range(i + 1, len(courses)):
            code_a, sched_a = courses[i]
            code_b, sched_b = courses[j]

            overlap = _find_time_overlap(sched_a, sched_b)
            if overlap:
                conflicts.append({
                    "course_a": code_a,
                    "course_b": code_b,
                    "overlap": overlap,
                })

    return conflicts


def _find_time_overlap(sched_a: dict, sched_b: dict) -> str | None:
    """
    Verifica sobreposição entre dois horários.
    Retorna descrição do conflito ou None se não houver.
    """
    days_a = set(sched_a.get("days", []))
    days_b = set(sched_b.get("days", []))
    common_days = days_a & days_b

    if not common_days:
        return None

    # Compara faixas de horário nos dias em comum
    times_a = sched_a.get("times", [])
    times_b = sched_b.get("times", [])

    for day in common_days:
        # Encontra os horários correspondentes a esse dia
        day_idx_a = _day_index(sched_a.get("days", []), day)
        day_idx_b = _day_index(sched_b.get("days", []), day)

        time_a = times_a[day_idx_a] if day_idx_a < len(times_a) else None
        time_b = times_b[day_idx_b] if day_idx_b < len(times_b) else None

        if time_a and time_b and _times_overlap(time_a, time_b):
            return f"conflito na {day}-feira: {time_a} × {time_b}"

    return None


def _day_index(days: list[str], target: str) -> int:
    try:
        return days.index(target)
    except ValueError:
        return 999


def _times_overlap(t1: str, t2: str) -> bool:
    """Verifica se duas faixas de horário se sobrepõem. Formato: 'HH:MM-HH:MM'."""
    try:
        def parse(t: str) -> tuple[int, int]:
            start, end = t.split("-")
            sh, sm = map(int, start.split(":"))
            eh, em = map(int, end.split(":"))
            return sh * 60 + sm, eh * 60 + em

        s1, e1 = parse(t1)
        s2, e2 = parse(t2)
        return s1 < e2 and s2 < e1  # sobreposição se não são disjuntos
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fluxo 3 — Gerador de trilha de estudos
# ---------------------------------------------------------------------------

def _run_trail(query: str, chunks: list[dict], llm: BaseChatModel) -> dict:
    """
    Gera uma trilha de estudos recursiva para uma disciplina-alvo.
    Expande pré-requisitos recursivamente (máx 3 níveis) consultando o corpus.
    """
    logger.info("[automation/trail] iniciando")

    # 1. Extrai a disciplina-alvo
    target = _extract_target_course(query, llm)
    if not target:
        return _error_result(
            "Não consegui identificar a disciplina-alvo. "
            "Exemplo: 'Quero chegar em COMP4501 (IA). O que devo estudar antes?'"
        )

    # 2. Expande pré-requisitos recursivamente
    all_sources: list[dict] = []
    trail = _expand_trail(target, llm, all_sources, depth=0, max_depth=3, visited=set())

    # 3. Gera sugestão de conteúdo de estudo para cada nó da trilha
    trail_with_content = _add_study_content(trail, llm, all_sources)

    # 4. Serializa para markdown
    summary = f"Trilha de estudos para chegar em **{target}** ({len(trail_with_content)} etapas):"
    details = _render_trail(trail_with_content)

    return {
        "type": "trail",
        "summary": summary,
        "details": details,
        "raw": trail_with_content,
        "sources": _deduplicate_sources(all_sources),
        "blocked": False,
        "block_reason": "",
    }


def _extract_target_course(query: str, llm: BaseChatModel) -> str:
    """Extrai o código/nome da disciplina-alvo da query."""
    # Regex para código no formato COMPXXXX
    codes = re.findall(r"\b[A-Z]{2,6}\d{3,5}\b", query)
    if codes:
        return codes[-1]  # geralmente o último código mencionado é o alvo

    # LLM fallback
    prompt = f"""\
Qual é a disciplina-alvo na mensagem abaixo? (a que o aluno quer chegar)
Retorne SOMENTE o código ou nome da disciplina, sem texto adicional.
Se não encontrar, retorne "".

Mensagem: "{query}"
"""
    try:
        response = llm.invoke(prompt)
        return response.content.strip().strip('"').strip()
    except Exception:
        return ""


def _expand_trail(
    course: str,
    llm: BaseChatModel,
    all_sources: list[dict],
    depth: int,
    max_depth: int,
    visited: set[str],
) -> list[dict]:
    """
    Recursivamente expande pré-requisitos de uma disciplina.
    Retorna lista de nós: [{course, level, prerequisites, study_hint}]
    """
    if depth > max_depth or course in visited:
        return []

    visited.add(course)
    course_chunks = retrieve_by_course_code(course, top_k=4)
    all_sources.extend(course_chunks)

    prereqs = _extract_prereqs_from_chunks(course, course_chunks, llm)

    node = {
        "course": course,
        "level": depth,
        "prerequisites": prereqs,
        "study_hint": "",
    }

    children = []
    for prereq in prereqs:
        children.extend(
            _expand_trail(prereq, llm, all_sources, depth + 1, max_depth, visited)
        )

    # Retorna em ordem topológica: pré-requisitos primeiro
    return children + [node]


def _add_study_content(trail: list[dict], llm: BaseChatModel, chunks: list[dict]) -> list[dict]:
    """
    Para cada disciplina na trilha, sugere tópicos de estudo preparatório
    com base nos chunks recuperados.
    """
    # Mapa course → chunks disponíveis
    chunk_map: dict[str, list[dict]] = {}
    for chunk in chunks:
        code = chunk.get("course_code", "")
        if code:
            chunk_map.setdefault(code, []).append(chunk)

    for node in trail:
        course = node["course"]
        relevant = chunk_map.get(course, [])

        if relevant:
            evidence = "\n".join(c.get("text", "")[:300] for c in relevant[:2])
            prompt = f"""\
Com base na ementa/descrição da disciplina "{course}" abaixo, sugira em 1–2 frases
o que o aluno deve estudar ou revisar antes de cursá-la.
Seja direto e prático. Não repita o nome da disciplina.

Ementa: {evidence}
"""
            try:
                response = llm.invoke(prompt)
                node["study_hint"] = response.content.strip()
            except Exception:
                node["study_hint"] = ""
        else:
            node["study_hint"] = ""

    return trail


def _render_trail(trail: list[dict]) -> list[str]:
    """Converte a trilha em linhas de markdown para exibição."""
    if not trail:
        return ["Não foi possível montar a trilha com os documentos disponíveis."]

    lines = []
    for i, node in enumerate(trail, 1):
        indent = "  " * node.get("level", 0)
        course = node["course"]
        hint = node.get("study_hint", "")
        prereqs = node.get("prerequisites", [])

        line = f"{indent}{i}. **{course}**"
        if prereqs:
            line += f" ← requer: {', '.join(prereqs)}"
        if hint:
            line += f"\n{indent}   _{hint}_"
        lines.append(line)

    return lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deduplicate_sources(chunks: list[dict]) -> list[dict]:
    """Remove chunks duplicados para a seção de fontes."""
    seen: set[str] = set()
    result = []
    for chunk in chunks:
        key = f"{chunk.get('source','')}:{chunk.get('page','')}"
        if key not in seen:
            seen.add(key)
            result.append({
                "source": chunk.get("source", ""),
                "page": chunk.get("page"),
                "excerpt": chunk.get("excerpt", ""),
            })
    return result


def _error_result(message: str) -> dict:
    return {
        "type": "error",
        "summary": message,
        "details": [],
        "raw": {},
        "sources": [],
        "blocked": False,
        "block_reason": "",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    tests = [
        ("prereq",   "Posso cursar COMP3501 se já fiz COMP2401 e COMP2201?"),
        ("schedule", "COMP3501 e MAT2001 têm conflito de horário?"),
        ("trail",    "Quero chegar em COMP4501 (Inteligência Artificial). O que estudar?"),
    ]

    atype = sys.argv[1] if len(sys.argv) > 1 else None
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None

    if atype and query:
        result = run_automation(atype, query)
        print(f"\nTipo: {result['type']}")
        print(f"Resumo: {result['summary']}")
        for d in result.get("details", []):
            print(f"  {d}")
    else:
        for t, q in tests:
            print(f"\n{'='*60}")
            print(f"Tipo: {t}  |  Query: {q}")
            result = run_automation(t, q)
            print(f"Resumo: {result['summary']}")
            for d in result.get("details", [])[:3]:
                print(f"  {d}")