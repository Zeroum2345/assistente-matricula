# app/streamlit_app.py
# Interface Streamlit do Assistente de Matrícula UFCG
#
# Funcionalidades:
#   - Chat com histórico de conversa
#   - Painel lateral com atalhos de automação
#   - Exibição de fontes/citações em expansor colapsável
#   - Indicador visual do fluxo percorrido no grafo (intent + self-check score)
#   - Modo debug opcional (mostra state completo do LangGraph)

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import streamlit as st

# Garante que o root do projeto está no path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Assistente de Matrícula UFCG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS customizado (mínimo, sem sobrescrever o tema)
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.source-chip {
    display: inline-block;
    padding: 2px 8px;
    background: rgba(100,100,200,0.12);
    border-radius: 12px;
    font-size: 0.78rem;
    margin: 2px 2px;
    color: inherit;
}
.intent-badge {
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-qa         { background:#1D9E7522; color:#0F6E56; }
.badge-automation { background:#BA751722; color:#854F0B; }
.badge-refuse     { background:#99211D22; color:#71130F; }
.score-ok   { color: #0F6E56; font-weight: 600; }
.score-warn { color: #854F0B; font-weight: 600; }
.score-fail { color: #71130F; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------

def _init_session() -> None:
    defaults = {
        "messages":     [],   # list[HumanMessage | AIMessage]
        "ui_history":   [],   # list[dict] para renderização (role, content, meta)
        "graph":        None,
        "debug_mode":   False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# ---------------------------------------------------------------------------
# Carrega o grafo (singleton por sessão)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando modelos (primeira vez pode demorar)...")
def _load_graph():
    from src.graph import get_compiled_graph
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎓 Assistente UFCG")
    st.caption("Matrícula · Pré-requisitos · Horários")

    st.divider()
    st.subheader("Automações rápidas")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Verificar\npré-requisitos", use_container_width=True):
            st.session_state["quick_query"] = (
                "Verifique os pré-requisitos para as disciplinas que quero cursar. "
                "Informe os códigos das disciplinas-alvo e as que já concluí."
            )
    with col2:
        if st.button("🗓️ Conflito\nde horário", use_container_width=True):
            st.session_state["quick_query"] = (
                "Verifique se há conflito de horário entre as disciplinas. "
                "Informe os códigos (ex: COMP3501, MAT2001)."
            )

    if st.button("📚 Trilha de estudos", use_container_width=True):
        st.session_state["quick_query"] = (
            "Gere uma trilha de estudos para chegar à disciplina-alvo. "
            "Qual disciplina você quer cursar?"
        )

    st.divider()
    st.subheader("Exemplos de perguntas")

    examples = [
        "Quais são os pré-requisitos de Redes de Computadores?",
        "Como funciona o trancamento de matrícula?",
        "Posso cursar COMP3501 se já fiz COMP2401 e COMP2201?",
        "COMP3501 e MAT2001 têm conflito de horário no 2025.1?",
        "Quero chegar em Inteligência Artificial. Que disciplinas devo fazer?",
        "Qual a diferença entre disciplina obrigatória e optativa?",
    ]

    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["quick_query"] = ex

    st.divider()
    st.session_state["debug_mode"] = st.toggle("🔧 Modo debug", value=False)

    st.divider()
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state["messages"]   = []
        st.session_state["ui_history"] = []
        st.rerun()

    # Estatísticas do índice
    st.divider()
    st.caption("Índice FAISS")
    try:
        from src.agents.retriever import get_index_stats
        stats = get_index_stats()
        if "error" in stats:
            st.warning("Índice não encontrado.\nExecute: `python ingest/indexer.py`")
        else:
            st.metric("Vetores", f"{stats['total_vectors']:,}")
            st.caption(f"Modelo: {stats['embedding_model']}")
    except Exception:
        st.warning("Não foi possível carregar estatísticas do índice.")


# ---------------------------------------------------------------------------
# Área principal
# ---------------------------------------------------------------------------

st.title("Assistente de Matrícula UFCG")
st.caption(
    "Pergunte sobre regulamentos, pré-requisitos e horários — "
    "ou use as automações na barra lateral."
)

def _render_meta(meta: dict) -> None:
    """Renderiza intent badge, self-check score e fontes colapsáveis."""

    cols = st.columns([1, 1, 4])

    # Intent badge
    intent = meta.get("intent", "")
    badge_class = {
        "qa":         "badge-qa",
        "automation": "badge-automation",
        "refuse":     "refuse",
    }.get(intent, "badge-qa")
    label = {"qa": "Q&A", "automation": "Automação", "refuse": "Recusa"}.get(intent, intent)

    with cols[0]:
        st.markdown(
            f'<span class="intent-badge {badge_class}">{label}</span>',
            unsafe_allow_html=True,
        )

    # Self-check score
    score = meta.get("self_check_score")
    if score is not None:
        css = "score-ok" if score >= 0.7 else ("score-warn" if score >= 0.5 else "score-fail")
        with cols[1]:
            st.markdown(
                f'<span class="{css}">✓ {score:.0%}</span>',
                unsafe_allow_html=True,
            )

    # Latência
    latency = meta.get("latency_ms")
    if latency:
        with cols[2]:
            st.caption(f"⏱ {latency}ms")

    # Fontes
    chunks = meta.get("retrieved_chunks", [])
    if chunks:
        with st.expander(f"📄 {len(chunks)} fonte(s) consultada(s)"):
            for c in chunks:
                source = c.get("source", "?")
                page   = c.get("page")
                score_c = c.get("score", 0)
                excerpt = c.get("excerpt", "")
                section = c.get("section", "")

                loc = f"pág. {page}" if page else ""
                sec = f" · {section}" if section else ""
                st.markdown(
                    f'<span class="source-chip">📎 {source} {loc}{sec} '
                    f'<small>({score_c:.2f})</small></span>',
                    unsafe_allow_html=True,
                )
                if excerpt:
                    st.caption(f"> {excerpt[:180]}…" if len(excerpt) > 180 else f"> {excerpt}")

    # Debug: state completo
    if st.session_state.get("debug_mode") and meta.get("full_state"):
        with st.expander("🔧 Estado completo do grafo"):
            import json
            state_display = {
                k: v for k, v in meta["full_state"].items()
                if k not in ("messages",)  # omite histórico completo
            }
            st.json(state_display)

# Renderiza histórico da conversa
for entry in st.session_state["ui_history"]:
    role    = entry["role"]
    content = entry["content"]
    meta    = entry.get("meta", {})

    with st.chat_message(role):
        st.markdown(content)

        # Mostra metadados do agente abaixo da resposta (somente assistant)
        if role == "assistant" and meta:
            _render_meta(meta)


# ---------------------------------------------------------------------------
# Input do usuário
# ---------------------------------------------------------------------------

# Aplica query rápida da sidebar se houver
quick = st.session_state.pop("quick_query", None)
query = st.chat_input("Faça uma pergunta sobre matrícula, pré-requisitos ou horários...") or quick

if query:
    # Mostra mensagem do usuário imediatamente
    st.session_state["ui_history"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Processa com o agente
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Processando…")

        try:
            graph = _load_graph()
            t0    = time.time()

            # Monta estado inicial
            from src.graph import AgentState
            history = [
                (HumanMessage if e["role"] == "user" else AIMessage)(content=e["content"])
                for e in st.session_state["ui_history"][:-1]  # exclui a mensagem atual
            ]
            history.append(HumanMessage(content=query))

            initial_state: AgentState = {
                "messages":          history,
                "query":             "",
                "intent":            "qa",
                "automation_type":   None,
                "retrieved_chunks":  [],
                "draft_answer":      "",
                "self_check_score":  0.0,
                "retry_count":       0,
                "final_answer":      "",
                "automation_result": None,
                "safety_blocked":    False,
                "safety_message":    "",
            }

            result = graph.invoke(
                initial_state,
                config={"recursion_limit": 10},
            )
            elapsed_ms = int((time.time() - t0) * 1000)

            answer = result.get("final_answer", "Não foi possível gerar uma resposta.")
            placeholder.markdown(answer)

            # Metadados para exibição
            meta = {
                "intent":            result.get("intent", "qa"),
                "automation_type":   result.get("automation_type"),
                "self_check_score":  result.get("self_check_score"),
                "retrieved_chunks":  result.get("retrieved_chunks", []),
                "latency_ms":        elapsed_ms,
                "full_state":        dict(result) if st.session_state["debug_mode"] else None,
            }
            _render_meta(meta)

        except FileNotFoundError as exc:
            answer = (
                "⚠️ Índice FAISS não encontrado. Execute primeiro:\n\n"
                "```bash\npython ingest/indexer.py\n```"
            )
            placeholder.error(answer)
            meta = {}

        except Exception as exc:
            logger.exception("[app] erro ao processar query")
            answer = f"❌ Erro interno: {exc}\n\nTente novamente ou consulte os logs."
            placeholder.error(answer)
            meta = {}

    # Salva no histórico
    st.session_state["messages"].append(HumanMessage(content=query))
    st.session_state["messages"].append(AIMessage(content=answer))
    st.session_state["ui_history"].append({
        "role":    "assistant",
        "content": answer,
        "meta":    meta,
    })