# =============================================================================
# Dockerfile — Assistente de Matrícula UFCG
# Multi-stage build: builder (deps) → runtime (app)
#
# Build:  docker build -t ufcg-agent .
# Run:    docker compose up          (recomendado, ver docker-compose.yml)
#         docker run -p 8501:8501 ufcg-agent  (standalone)
# =============================================================================

# =============================================================================
# Stage 1 — builder
# Instala Poetry e resolve dependências para uma venv isolada
# =============================================================================
FROM python:3.11-slim AS builder

# Evita prompts interativos durante apt
ENV DEBIAN_FRONTEND=noninteractive

# Dependências de sistema para compilar extensões nativas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instala Poetry (versão fixa para reprodutibilidade)
ENV POETRY_VERSION=1.8.3
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/tmp/poetry-cache

RUN python -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install --upgrade pip \
    && $POETRY_VENV/bin/pip install poetry==$POETRY_VERSION

ENV PATH="$POETRY_VENV/bin:$PATH"

WORKDIR /app

# Copia apenas os arquivos de dependências primeiro (melhor cache)
COPY pyproject.toml poetry.lock* ./

# Configura Poetry para não criar venv (instala no sistema do container)
RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction --no-ansi \
    && rm -rf $POETRY_CACHE_DIR

# =============================================================================
# Stage 2 — runtime
# Imagem final enxuta com apenas o necessário para rodar
# =============================================================================
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime deps: libgomp1 (FAISS), curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root para segurança
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copia pacotes Python instalados no builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia código-fonte
COPY --chown=appuser:appuser src/       ./src/
COPY --chown=appuser:appuser mcp/       ./mcp/
COPY --chown=appuser:appuser ingest/    ./ingest/
COPY --chown=appuser:appuser app/       ./app/
COPY --chown=appuser:appuser eval/      ./eval/
COPY --chown=appuser:appuser .env.example .env.example

# Cria diretórios de dados com permissão correta
RUN mkdir -p data/raw data/faiss_index logs \
    && chown -R appuser:appuser data logs

# Cria __init__.py nos pacotes (necessário para imports)
RUN for dir in src src/agents mcp mcp/mcp_docstore ingest eval; do \
        touch $dir/__init__.py 2>/dev/null || true; \
    done

USER appuser

# Variáveis de ambiente padrão (sobrescritas pelo docker-compose ou -e)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV RAW_DIR=/app/data/raw
ENV FAISS_INDEX_PATH=/app/data/faiss_index
ENV MCP_AUDIT_LOG=/app/logs/mcp_audit.jsonl
ENV EMBEDDING_MODEL=BAAI/bge-m3
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Porta do Streamlit
EXPOSE 8501

# Healthcheck: verifica se o Streamlit está respondendo
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entrypoint padrão: sobe a interface Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]