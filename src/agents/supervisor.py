# src/agents/supervisor.py
# Supervisor Agent — classifica a intent do usuário e determina o tipo de automação
#
# Responsabilidades:
#   1. Ler a query e decidir entre "qa", "automation" ou "refuse"
#   2. Se "automation", identificar o sub-tipo: "prereq", "schedule" ou "trail"
#   3. Retornar a classificação de forma determinística (temperature=0, saída estruturada)

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------

Intent = Literal["qa", "automation", "refuse"]
AutomationType = Literal["prereq", "schedule", "trail"] | None

# ---------------------------------------------------------------------------
# Prompt do Supervisor
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Você é o Supervisor de um assistente acadêmico da UFCG (Universidade Federal de Campina Grande).
Sua única função é classificar a intenção da mensagem do usuário em uma das três categorias abaixo.

=== CATEGORIAS ===

1. "qa"
   Perguntas sobre regulamentos, disciplinas, currículos, pré-requisitos (informativo),
   normas de matrícula, calendário acadêmico, créditos, equivalências.
   Exemplos:
   - "O que é crédito especial?"
   - "Quantos créditos preciso para me formar em CC?"
   - "Quais são os pré-requisitos de Cálculo 2?"
   - "Como funciona o trancamento de matrícula?"

2. "automation"
   Pedidos para EXECUTAR uma verificação, checagem ou geração de plano.
   Sub-tipos:
   - "prereq"   → verificar se o aluno PODE cursar uma ou mais disciplinas com base no que já cursou
   - "schedule" → detectar CONFLITO DE HORÁRIO entre disciplinas escolhidas
   - "trail"    → gerar TRILHA DE ESTUDOS / sequência recomendada para chegar a uma disciplina-alvo
   Exemplos:
   - "Posso cursar Redes se já fiz SO e Algoritmos?" → prereq
   - "Verifique se COMP1001, MAT2001 e COMP3002 batem no horário" → schedule
   - "Quero chegar em Compiladores, o que devo fazer antes?" → trail
   - "Me dê um plano para cursar IA" → trail

3. "refuse"
   Tudo que estiver FORA DO ESCOPO acadêmico da UFCG:
   - Perguntas pessoais, notas individuais, dados do SIGAA de um aluno específico
   - Assuntos não acadêmicos (clima, esportes, política, saúde, etc.)
   - Perguntas que não têm resposta nos documentos públicos da UFCG
   - Pedidos de conselho pessoal ("devo trancar o curso?")

=== REGRAS ESTRITAS ===
- Responda SOMENTE com um objeto JSON válido, sem texto adicional, sem markdown, sem explicações.
- Formato obrigatório:
  {"intent": "<qa|automation|refuse>", "automation_type": "<prereq|schedule|trail|null>"}
- "automation_type" deve ser null quando intent != "automation"
- Em caso de dúvida entre "qa" e "refuse", prefira "qa"
- Em caso de dúvida entre "qa" e "automation", prefira "automation" se houver verbo de ação (verificar, checar, gerar, listar conflitos, me dê um plano)
"""

_HUMAN_TEMPLATE = "Mensagem do usuário: {query}"

_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=_SYSTEM_PROMPT),
    HumanMessage(content=_HUMAN_TEMPLATE),
])

# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def classify_intent(
    query: str,
    llm: BaseChatModel,
) -> tuple[Intent, AutomationType]:
    """
    Classifica a intent da query usando o LLM.

    Args:
        query: Pergunta do usuário.
        llm:   Instância do ChatOllama (deve usar temperature=0).

    Returns:
        Tupla (intent, automation_type).
        automation_type é None quando intent != "automation".

    Raises:
        Nunca levanta exceção — em caso de falha de parsing retorna ("qa", None)
        como fallback seguro para não bloquear o fluxo.
    """
    if not query or not query.strip():
        logger.warning("[supervisor] query vazia → refuse")
        return "refuse", None

    # Aplica regras heurísticas rápidas antes de chamar o LLM
    fast_result = _fast_classify(query)
    if fast_result is not None:
        intent, automation_type = fast_result
        logger.info("[supervisor] fast-path → intent=%s  automation_type=%s", intent, automation_type)
        return intent, automation_type

    # Chama o LLM
    try:
        messages = _PROMPT.format_messages(query=query)
        response = llm.invoke(messages)
        raw = response.content.strip()
        logger.debug("[supervisor] LLM raw response: %r", raw)

        intent, automation_type = _parse_response(raw)
        logger.info("[supervisor] LLM → intent=%s  automation_type=%s", intent, automation_type)
        return intent, automation_type

    except Exception as exc:
        logger.error("[supervisor] erro ao chamar LLM: %s — fallback para 'qa'", exc)
        return "qa", None


# ---------------------------------------------------------------------------
# Parser da resposta do LLM
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> tuple[Intent, AutomationType]:
    """
    Extrai intent e automation_type do JSON retornado pelo LLM.
    Tenta parsing direto; se falhar, tenta extrair via regex.
    """
    # Limpa possíveis artefatos de markdown (```json ... ```)
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: tenta capturar os valores com regex
        data = _regex_extract(cleaned)

    intent_raw = str(data.get("intent", "qa")).lower().strip()
    auto_raw = str(data.get("automation_type", "null")).lower().strip()

    # Valida e normaliza intent
    if intent_raw not in ("qa", "automation", "refuse"):
        logger.warning("[supervisor] intent inválida %r → fallback 'qa'", intent_raw)
        intent_raw = "qa"

    # Valida e normaliza automation_type
    if intent_raw == "automation":
        if auto_raw not in ("prereq", "schedule", "trail"):
            logger.warning("[supervisor] automation_type inválido %r → fallback 'trail'", auto_raw)
            auto_raw = "trail"
        automation_type: AutomationType = auto_raw  # type: ignore[assignment]
    else:
        automation_type = None

    return intent_raw, automation_type  # type: ignore[return-value]


def _regex_extract(text: str) -> dict:
    """Extrai campos via regex quando o JSON está malformado."""
    intent_match = re.search(r'"intent"\s*:\s*"(\w+)"', text)
    auto_match = re.search(r'"automation_type"\s*:\s*"(\w+)"', text)

    intent = intent_match.group(1) if intent_match else "qa"
    automation_type = auto_match.group(1) if auto_match else "null"

    logger.debug("[supervisor] regex fallback → intent=%r  automation_type=%r", intent, automation_type)
    return {"intent": intent, "automation_type": automation_type}


# ---------------------------------------------------------------------------
# Classificador heurístico rápido (evita latência do LLM para casos óbvios)
# ---------------------------------------------------------------------------

# Palavras-chave que indicam pedido de automação por tipo
_SCHEDULE_KEYWORDS = [
    "conflito de horário", "conflito horário", "batem no horário", "horário bate",
    "chocam", "sobreposição de horário", "verificar horário", "checar horário",
    "horários conflitam",
]

_PREREQ_KEYWORDS = [
    "posso cursar", "consigo cursar", "tenho pré-requisito", "tenho prereq",
    "já fiz os pré", "já cumpri", "pré-requisitos satisfeitos", "pré-requisito ok",
    "verificar pré-req", "checar pré-req",
]

_TRAIL_KEYWORDS = [
    "trilha de estudos", "plano de estudos", "o que estudar antes", "sequência de disciplinas",
    "me dê um plano", "quero chegar em", "como chegar em", "caminho até",
    "o que devo fazer antes", "roadmap", "o que preciso antes",
]

_REFUSE_KEYWORDS = [
    "minha nota", "meu histórico", "meu sigaa", "devo trancar", "devo desistir",
    "me ajuda a decidir se", "qual curso fazer", "futebol", "política", "clima",
]


def _fast_classify(query: str) -> tuple[Intent, AutomationType] | None:
    """
    Classificação heurística rápida por palavras-chave.
    Retorna None se não for possível classificar com confiança
    (delegando ao LLM).
    """
    q = query.lower()

    for kw in _REFUSE_KEYWORDS:
        if kw in q:
            return "refuse", None

    for kw in _SCHEDULE_KEYWORDS:
        if kw in q:
            return "automation", "schedule"

    for kw in _PREREQ_KEYWORDS:
        if kw in q:
            return "automation", "prereq"

    for kw in _TRAIL_KEYWORDS:
        if kw in q:
            return "automation", "trail"

    # Não conseguiu classificar com confiança → delega ao LLM
    return None


# ---------------------------------------------------------------------------
# Testes rápidos (python -m src.agents.supervisor)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="qwen2.5:3b", temperature=0.0)

    test_cases = [
        # (query, intent_esperada, automation_type_esperado)
        ("Quais são os pré-requisitos de Redes de Computadores?",          "qa",         None),
        ("Como funciona o trancamento de matrícula na UFCG?",              "qa",         None),
        ("Posso cursar Compiladores se já fiz LLP e Estruturas?",          "automation", "prereq"),
        ("Verifique se COMP1001 e MAT2001 têm conflito de horário",        "automation", "schedule"),
        ("Quero um plano para chegar em Inteligência Artificial",          "automation", "trail"),
        ("Quanto custou o bandejão hoje?",                                 "refuse",     None),
        ("Qual é a minha nota em Cálculo 1?",                              "refuse",     None),
    ]

    query = sys.argv[1] if len(sys.argv) > 1 else None

    if query:
        intent, auto_type = classify_intent(query, llm)
        print(f"intent={intent}  automation_type={auto_type}")
    else:
        print(f"{'Query':<55} {'Esperado':<22} {'Obtido':<22} {'OK?'}")
        print("-" * 110)
        for q, exp_intent, exp_auto in test_cases:
            intent, auto_type = classify_intent(q, llm)
            got = f"{intent} / {auto_type}"
            exp = f"{exp_intent} / {exp_auto}"
            ok = "✅" if intent == exp_intent and auto_type == exp_auto else "❌"
            print(f"{q:<55} {exp:<22} {got:<22} {ok}")