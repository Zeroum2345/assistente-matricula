# 🎓 Assistente de Matrícula UFCG

> Sistema agêntico open source para auxiliar estudantes da UFCG em decisões de matrícula:
> verificação de pré-requisitos, detecção de conflitos de horário e geração de trilhas de estudo.
> Construído com **LangGraph**, **FAISS**, **Qwen 2.5 (Ollama)** e **MCP**.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Eureca API](https://img.shields.io/badge/Eureca-UFCG-orange.svg)](https://eureca.lsd.ufcg.edu.br)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Visão Geral

Estudantes da UFCG enfrentam três dores no período de matrícula: descobrem tarde que não têm pré-requisitos, percebem conflitos de horário só após montar a grade, e não sabem quais disciplinas estudar para chegar a uma disciplina avançada. As informações existem em PDFs dispersos — este projeto as unifica em um assistente conversacional.

### Stack

| Componente | Tecnologia |
|---|---|
| LLM | Qwen 2.5 via Ollama (local) |
| Embeddings | `BAAI/bge-m3` ou `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector store | FAISS local |
| Orquestração | LangGraph `StateGraph` |
| Interface | Streamlit |
| MCP | `mcp-docstore` (corpus) + `mcp-eureca` (API UFCG) |
| Avaliação | RAGAS |

---

## Estrutura

```
ufcg-matricula-agent/
├── src/
│   ├── graph.py                  # LangGraph StateGraph
│   ├── agents/
│   │   ├── supervisor.py         # Classifica intent (qa/automation/refuse)
│   │   ├── retriever.py          # FAISS + embeddings + MMR
│   │   ├── self_check.py         # Self-RAG (heurística para 3B, LLM para 7B+)
│   │   ├── safety.py             # Disclaimers e bloqueio de escopo
│   │   ├── answerer.py           # Gera resposta com citações
│   │   ├── automation.py         # Pré-req, conflito, trilha via FAISS
│   │   └── automation_eureca.py  # Mesmas automações + fallback Eureca → FAISS
│   └── integrations/
│       └── eureca_client.py      # Cliente HTTP da API Eureca/SIGAA
├── mcp/
│   ├── mcp_docstore/server.py    # MCP: corpus estático (FAISS)
│   └── mcp_eureca/server.py      # MCP: API Eureca tempo real
├── ingest/indexer.py             # PDF/HTML → chunks → FAISS
├── app/streamlit_app.py          # Interface
├── eval/run_ragas.py             # Avaliação RAGAS
├── data/raw/                     # ← PDFs da UFCG aqui
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Instalação rápida

### Com Docker (recomendado)

```bash
# 1. Sobe o Ollama
docker compose up ollama -d

# 2. Baixa o modelo (2 GB para 3B, 5 GB para 7B)
docker compose run --rm ollama-pull

# 3. Coloca os PDFs em data/raw/ e indexa
docker compose run --rm ingest

# 4. Sobe a interface → http://localhost:8501
docker compose up agent -d
```

> **RAM necessária:** 4 GB para `qwen2.5:3b`, 8 GB para `qwen2.5:7b`.
> No Docker Desktop: **Settings → Resources → Memory**.

### Com Poetry (desenvolvimento)

```bash
poetry install --with eval,dev
cp .env.example .env

# Cria estrutura de pastas e __init__.py
mkdir -p data/raw data/faiss_index logs src/agents src/integrations mcp/mcp_docstore mcp/mcp_eureca ingest eval
touch src/__init__.py src/agents/__init__.py src/integrations/__init__.py
touch mcp/__init__.py mcp/mcp_docstore/__init__.py mcp/mcp_eureca/__init__.py
touch ingest/__init__.py eval/__init__.py

# Indexa documentos e sobe
ollama pull qwen2.5:7b
PYTHONPATH=. poetry run python ingest/indexer.py --reset
PYTHONPATH=. poetry run streamlit run app/streamlit_app.py --server.fileWatcherType none
```

---

## Configuração (`.env`)

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434   # Docker: http://ollama:11434
OLLAMA_MODEL=qwen2.5:7b                 # ou qwen2.5:3b para menos RAM

# Embeddings (troque e reindexe se mudar)
EMBEDDING_MODEL=BAAI/bge-m3
# EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  # para pouca RAM

# Self-check via LLM (true para 7B+, false para 3B)
SELF_CHECK_USE_LLM=true

# API Eureca (opcional — sem credenciais usa apenas FAISS)
EURECA_LOGIN=
EURECA_PASSWD=
EURECA_API_URL=https://eureca.lsd.ufcg.edu.br/das/v2
EURECA_PERIODO_PADRAO=2025.1
EURECA_CURRICULO_PADRAO=14102100
```

---

## Documentos para indexar

Coloque em `data/raw/` — formatos aceitos: `.pdf`, `.html`, `.txt`.

| Documento | Onde encontrar |
|---|---|
| PPC do curso (pré-requisitos, ementas) | Site da unidade acadêmica |
| Regulamento dos Cursos de Graduação | `portal.ufcg.edu.br` → Ensino → Regulamentos |
| Calendário Acadêmico | `portal.ufcg.edu.br` → Ensino → Calendário |

---

## Automações disponíveis

| Tipo | Trigger | Exemplo |
|---|---|---|
| **Pré-requisitos** | "posso cursar", "tenho os pré-req" | *"Posso cursar Redes se já fiz SO?"* |
| **Conflito de horário** | "conflito de horário", "batem no horário" | *"COMP3501 e MAT2001 têm conflito no 2025.1?"* |
| **Trilha de estudos** | "quero chegar em", "trilha de estudos" | *"Que disciplinas devo fazer para chegar em IA?"* |

---

## MCP Servers

Dois servidores MCP próprios expõem os dados como ferramentas padronizadas com allowlist, sanitização de input e audit log em JSONL.

**`mcp-docstore`** — corpus estático (FAISS): `search_docs`, `get_prerequisites`, `get_schedule`
**`mcp-eureca`** — API Eureca tempo real: `get_prerequisitos_eureca`, `get_horarios_eureca`, `verificar_conflito_eureca`, `get_turmas_eureca`

As automações tentam o `mcp-eureca` primeiro e caem para o FAISS se a API não estiver disponível.

---

## Self-RAG (anti-alucinação)

O `self_check.py` valida se o rascunho gerado está suportado pelos chunks recuperados antes de entregar a resposta:

- **Modelos 3B/1B:** heurística por sobreposição de vocabulário (sem chamar o LLM).
- **Modelos 7B+:** avaliação via LLM com JSON estruturado.

Se o score for < 0.7, o grafo faz uma re-busca com query expandida (máximo 1 retry). Se reprovar novamente, recusa educadamente.

---

## Avaliação

```bash
PYTHONPATH=. poetry run python eval/run_ragas.py --suite rag
PYTHONPATH=. poetry run python eval/run_ragas.py --suite automation
PYTHONPATH=. poetry run python eval/run_ragas.py --suite mcp
```

| Suite | Métricas |
|---|---|
| RAG (20 queries) | context_precision, context_recall, faithfulness, answer_relevancy, latência |
| Automação (5 tarefas) | taxa de sucesso, tempo médio por tipo |
| MCP (3 tools) | disponibilidade, latência, entradas no audit log |

---

## Solução de problemas comuns

**`ModuleNotFoundError: No module named 'src'`** — rode sempre com `PYTHONPATH=. poetry run ...`

**`ValueError: 'final_answer' is already being used as a state key`** — todos os nós do grafo devem ter prefixo `node_` (ex: `node_final_answer`).

**Segfault ao carregar bge-m3 no macOS** — force CPU em `retriever.py` e troque para o modelo MiniLM no `.env`.

**Self-check sempre 0%** — defina `OLLAMA_MODEL=qwen2.5:3b` no `.env` ou `SELF_CHECK_USE_LLM=false` para forçar a heurística.

**Ollama cai quando o agent sobe** — RAM insuficiente. Aumente para 8 GB no Docker Desktop ou use `qwen2.5:3b`.

---

## Segurança

- MCP servers com allowlist de tools, acesso somente leitura e logs de auditoria em `logs/`.
- Credenciais Eureca armazenadas apenas em memória (nunca em disco ou logs).
- Safety Agent bloqueia queries sobre notas pessoais, credenciais e conselhos de vida acadêmica.
- Corpus indexado contém apenas documentos públicos da UFCG.

---

## Referências

- Asai et al. (2023). [Self-RAG](https://arxiv.org/abs/2310.11511). *arXiv*.
- Es et al. (2023). [RAGAS](https://arxiv.org/abs/2309.15217). *arXiv*.
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Model Context Protocol](https://modelcontextprotocol.io/specification)
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

---

## Licença

MIT — veja [LICENSE](LICENSE).

```bibtex
@software{assistente_matricula_2026,
  author  = {Marcus Paulo dos Santos Ferreira},
  title   = {Assistente de Matrícula UFCG},
  year    = {2026},
  url     = {https://github.com/Zeroum2345/assistente-matricula},
  license = {MIT}
}
```