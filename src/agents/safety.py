# src/agents/safety.py
# Safety / Policy Agent — adiciona disclaimers e bloqueia respostas fora do escopo
#
# Responsabilidades:
#   1. Detectar se a query/resposta envolve conselho pessoal ou fora do escopo
#   2. Adicionar disclaimer padrão quando o conteúdo for sensível (ex: regulamentos)
#   3. Bloquear completamente respostas que violem a política de escopo
#   4. Nunca modificar o conteúdo factual — apenas adicionar avisos

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disclaimer padrão (sempre incluído em respostas sobre regulamentos)
# ---------------------------------------------------------------------------

DISCLAIMER_PADRAO = (
    "As informações acima são baseadas nos documentos públicos da UFCG "
    "disponíveis no momento da indexação. Para confirmação oficial, consulte "
    "o DAA (Departamento de Administração Acadêmica) ou o SIGAA."
)

DISCLAIMER_PREREQ = (
    "A verificação de pré-requisitos é baseada nos fluxogramas e regulamentos "
    "indexados. A situação real da sua matrícula deve ser confirmada no SIGAA."
)

DISCLAIMER_HORARIO = (
    "Os horários verificados são baseados nos dados indexados e podem não "
    "refletir alterações de última hora. Confirme no SIGAA antes de se matricular."
)

# ---------------------------------------------------------------------------
# Padrões que sempre bloqueiam (escopo fora do acadêmico da UFCG)
# ---------------------------------------------------------------------------

_BLOCK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(minha nota|meu histórico|minha situação acadêmica|"
            r"devo trancar|devo desistir|devo mudar de curso|"
            r"me ajuda a decidir|conselho pessoal)\b",
            re.IGNORECASE,
        ),
        "Não posso oferecer conselho pessoal sobre decisões acadêmicas individuais. "
        "Procure o DAA ou o serviço de orientação acadêmica da UFCG.",
    ),
    (
        re.compile(
            r"\b(senha|login|acesso ao sigaa|meu cadastro|minha conta)\b",
            re.IGNORECASE,
        ),
        "Não posso ajudar com credenciais ou acesso a sistemas da UFCG. "
        "Entre em contato com a STI (Superintendência de Tecnologia da Informação).",
    ),
    (
        re.compile(
            r"\b(dado pessoal|cpf|matrícula do aluno|número de matrícula)\b",
            re.IGNORECASE,
        ),
        "Não tenho acesso a dados pessoais de estudantes. "
        "Consulte o SIGAA com suas credenciais.",
    ),
]

# ---------------------------------------------------------------------------
# Padrões que adicionam disclaimer (sem bloquear)
# ---------------------------------------------------------------------------

_DISCLAIMER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(pré.?requisito|prereq|fluxograma|currículo|grade curricular|"
            r"disciplina obrigatória|disciplina optativa|crédito)\b",
            re.IGNORECASE,
        ),
        DISCLAIMER_PREREQ,
    ),
    (
        re.compile(
            r"\b(horário|turno|sala|turma|conflito de horário|período letivo)\b",
            re.IGNORECASE,
        ),
        DISCLAIMER_HORARIO,
    ),
    (
        re.compile(
            r"\b(regulamento|resolução|norma|política acadêmica|prazo|calendário)\b",
            re.IGNORECASE,
        ),
        DISCLAIMER_PADRAO,
    ),
]


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def apply_safety(
    query: str,
    draft: str = "",
    automation_result: dict | None = None,
) -> tuple[bool, str]:
    """
    Verifica a política de segurança e escopo.

    Args:
        query:             Pergunta original do usuário.
        draft:             Rascunho de resposta gerado pelo Answerer (pode ser vazio).
        automation_result: Resultado de automação (se houver).

    Returns:
        (blocked, message)
        - blocked=True:  a resposta deve ser bloqueada; message explica o motivo.
        - blocked=False: a resposta pode prosseguir; message é um disclaimer (pode ser "").
    """
    # Texto combinado para análise
    combined = f"{query} {draft}".strip()

    # 1. Verifica padrões de bloqueio (na query — não queremos bloquear por conteúdo do draft)
    for pattern, reason in _BLOCK_PATTERNS:
        if pattern.search(query):
            logger.warning("[safety] bloqueado — padrão: %r", pattern.pattern[:50])
            return True, reason

    # 2. Verifica se a automação tem resultado de bloqueio interno
    if automation_result and automation_result.get("blocked"):
        reason = automation_result.get("block_reason", "Resultado fora do escopo permitido.")
        logger.warning("[safety] bloqueado pelo automation_result: %s", reason)
        return True, reason

    # 3. Adiciona disclaimer apropriado (sem bloquear)
    for pattern, disclaimer in _DISCLAIMER_PATTERNS:
        if pattern.search(combined):
            logger.info("[safety] disclaimer adicionado — padrão: %r", pattern.pattern[:50])
            return False, disclaimer

    # 4. Sem disclaimers específicos: adiciona o padrão geral como boa prática
    return False, DISCLAIMER_PADRAO


# ---------------------------------------------------------------------------
# Utilitário: injeta disclaimer no texto (usado pelo answerer)
# ---------------------------------------------------------------------------

def inject_disclaimer(text: str, disclaimer: str) -> str:
    """
    Adiciona o disclaimer ao final do texto, separado por uma linha horizontal.
    Evita duplicar se o disclaimer já estiver presente.
    """
    if not disclaimer or disclaimer in text:
        return text
    return f"{text}\n\n---\n> ℹ️ {disclaimer}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    tests = [
        ("Quais são os pré-requisitos de Cálculo 2?", "", False),
        ("Devo trancar meu curso este semestre?", "", True),
        ("Qual minha senha do SIGAA?", "", True),
        ("Quais são os horários de Redes no período 2025.1?", "", False),
        ("Como funciona o trancamento?", "", False),
    ]

    print(f"{'Query':<55} {'Esperado':<10} {'Obtido':<10} {'OK?'}")
    print("-" * 90)
    for q, draft, expected_block in tests:
        blocked, msg = apply_safety(q, draft)
        ok = "✅" if blocked == expected_block else "❌"
        print(f"{q:<55} {str(expected_block):<10} {str(blocked):<10} {ok}")
        if msg:
            print(f"  → {msg[:80]}")