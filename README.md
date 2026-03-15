# 🎓 Assistente de Matrícula UFCG

> Sistema agêntico open source para auxiliar estudantes da UFCG em decisões de matrícula:
> verificação de pré-requisitos, detecção de conflitos de horário e geração de trilhas de estudo.
> Construído com **LangChain + LangGraph**, **FAISS**, **bge-m3**, **Qwen 2.5 (Ollama)**, **MCP** e integração com a **API Eureca/SIGAA**.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Eureca API](https://img.shields.io/badge/Eureca-UFCG-orange.svg)](https://eureca.lsd.ufcg.edu.br)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Arquitetura do Sistema](#2-arquitetura-do-sistema)
3. [Estrutura do Repositório](#3-estrutura-do-repositório)
4. [Pré-requisitos](#4-pré-requisitos)
5. [Instalação — Poetry (desenvolvimento)](#5-instalação--poetry-desenvolvimento)
6. [Instalação — Docker (produção/demo)](#6-instalação--docker-produçãodemo)
7. [Configuração](#7-configuração)
8. [Ingestão de Documentos](#8-ingestão-de-documentos)
9. [Executando o Sistema](#9-executando-o-sistema)
10. [Usando a Interface](#10-usando-a-interface)
11. [Agentes em Detalhe](#11-agentes-em-detalhe)
12. [MCP Servers](#12-mcp-servers)
13. [Integração com a API Eureca](#13-integração-com-a-api-eureca)
14. [Automações Disponíveis](#14-automações-disponíveis)
15. [Mecanismo Anti-Alucinação (Self-RAG)](#15-mecanismo-anti-alucinação-self-rag)
16. [Avaliação](#16-avaliação)
17. [Segurança](#17-segurança)
18. [Solução de Problemas](#18-solução-de-problemas)
19. [Limitações e Próximos Passos](#19-limitações-e-próximos-passos)
20. [Referências](#20-referências)

---

## 1. Visão Geral

### Problema

Estudantes da UFCG frequentemente enfrentam dificuldades no período de matrícula:

- **Pré-requisitos**: descobrem tarde que não podem cursar uma disciplina por falta de pré-requisito.
- **Conflitos de horário**: só percebem sobreposições depois de feita a matrícula.
- **Planejamento**: não sabem quais disciplinas estudar para chegar a uma disciplina-alvo avançada.

Essas informações existem em PDFs e páginas dispersas do site da UFCG, mas são difíceis de consultar rapidamente.

### Solução

Um assistente conversacional que:

1. **Responde perguntas** sobre regulamentos, disciplinas e normas acadêmicas com **citações obrigatórias** dos documentos originais.
2. **Executa automações** como verificar pré-requisitos, detectar conflitos de horário e gerar trilhas de estudo.
3. **Nunca inventa**: implementa Self-RAG para validar cada afirmação contra as evidências recuperadas antes de responder.

### Tecnologias principais

| Componente | Tecnologia |
|---|---|
| LLM | Qwen 2.5 7B via Ollama (local, sem SaaS) |
| Embeddings | `BAAI/bge-m3` (multilingual, ótimo para PT-BR) |
| Vector store | FAISS (local, sem SaaS) |
| Orquestração | LangGraph `StateGraph` |
| Interface | Streamlit |
| MCP (corpus) | `fastmcp` — servidor próprio `mcp-docstore` |
| MCP (tempo real) | `fastmcp` — servidor próprio `mcp-eureca` |
| API externa | Eureca UFCG (`eureca.lsd.ufcg.edu.br/das/v2`) |
| Avaliação | RAGAS |

---

## 2. Arquitetura do Sistema

### Grafo de agentes (LangGraph)

```
Usuário
   │
   ▼
[entry] ──────────────────────────────────────────────
   │         extrai query da mensagem
   ▼
[supervisor] ──────────────────────────────────────────
   │         classifica intent: qa / automation / refuse
   ├──── "refuse" ────────────────────────────► [refuse] ──► END
   │
   ▼  (qa ou automation)
[retriever] ───────────────────────────────────────────
   │         FAISS + bge-m3, top-k chunks com MMR
   │
   ├──── intent="automation" ─────────────────► [automation]
   │                                                │
   │         (prereq / schedule / trail)            │
   │                                                │
   └──── intent="qa" ──────────────────────────────►│
                                                     ▼
                                               [safety]
                                                 │
                                    ┌────────────┤
                                    │            │
                               "blocked"        "ok"
                                    │            │
                                    │            ▼
                                    │       [answerer]
                                    │            │
                                    │            ▼
                                    │       [self_check]
                                    │            │
                                    │     ┌──────┼──────────────┐
                                    │  score     │          score
                                    │  ≥ 0.7  retry(1x)    < 0.7
                                    │     │      │          2ª vez
                                    │     │   [retriever]      │
                                    │     │   (query expandida) │
                                    │     │                     ▼
                                    │     │               [refuse]
                                    │     │
                                    ▼     ▼
                              [final_answer] ──► END
```

### Fluxo de uma pergunta típica

**Exemplo:** *"Posso cursar Redes de Computadores se já fiz SO?"*

1. `entry` extrai a query da última mensagem.
2. `supervisor` classifica: `intent="automation"`, `automation_type="prereq"`.
3. `retriever` busca chunks sobre pré-requisitos de Redes no FAISS com MMR.
4. `automation` executa `_run_prereq`: extrai cursos da query, busca pré-requisitos no corpus, compara com o que o aluno já cursou.
5. `safety` adiciona disclaimer: *"confirme no SIGAA"*.
6. `answerer` formata a resposta em markdown com citações `[Fonte: fluxograma_cc.pdf, pág. 5]`.
7. `self_check` avalia se as afirmações estão suportadas pelos chunks → score ≥ 0.7 → aprova.
8. `final_answer` consolida e retorna ao usuário.

---

## 3. Estrutura do Repositório

```
ufcg-matricula-agent/
│
├── src/                          # Código principal
│   ├── __init__.py
│   ├── graph.py                  # LangGraph StateGraph completo
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py         # Classifica intent (qa/automation/refuse)
│   │   ├── retriever.py          # FAISS + bge-m3 + MMR
│   │   ├── self_check.py         # Self-RAG: valida evidências
│   │   ├── safety.py             # Disclaimers e bloqueio de escopo
│   │   ├── answerer.py           # Gera resposta com citações obrigatórias
│   │   ├── automation.py         # Pré-req, conflito horário, trilha (FAISS)
│   │   └── automation_eureca.py  # Mesmas automações com fallback Eureca → FAISS
│   └── integrations/
│       ├── __init__.py
│       └── eureca_client.py      # Cliente HTTP da API Eureca/SIGAA
│
├── mcp/                          # MCP Servers próprios
│   ├── __init__.py
│   ├── mcp_docstore/
│   │   ├── __init__.py
│   │   └── server.py             # Corpus estático: search_docs, get_prerequisites, get_schedule
│   └── mcp_eureca/
│       ├── __init__.py
│       └── server.py             # API Eureca tempo real: horários, prereqs, turmas
│
├── ingest/                       # Pipeline de ingestão
│   ├── __init__.py
│   └── indexer.py                # PDF/HTML → chunks → FAISS
│
├── app/
│   └── streamlit_app.py          # Interface Streamlit
│
├── eval/
│   ├── run_ragas.py              # Avaliação: RAG + automação + MCP
│   ├── qa_dataset.json           # 20 perguntas rotuladas (gerado)
│   └── results.json              # Resultados da última avaliação (gerado)
│
├── tests/
│   ├── test_supervisor.py
│   ├── test_retriever.py
│   └── test_automation.py
│
├── data/
│   ├── raw/                      # ← Coloque os PDFs e HTMLs da UFCG aqui
│   └── faiss_index/              # Gerado pelo indexer (não versionar)
│
├── logs/
│   ├── mcp_audit.jsonl           # Log de auditoria do mcp-docstore (gerado)
│   └── mcp_eureca_audit.jsonl    # Log de auditoria do mcp-eureca (gerado)
│
├── pyproject.toml                # Dependências Poetry
├── poetry.lock                   # Lock file (versionar)
├── Dockerfile                    # Multi-stage build
├── docker-compose.yml            # Orquestração dos serviços
├── .env.example                  # Template de variáveis de ambiente
├── .env                          # Variáveis locais (NÃO versionar)
├── .gitignore
├── LICENSE
├── CITATION.cff
└── README.md
```

---

## 4. Pré-requisitos

### Sem Docker (desenvolvimento local)

| Requisito | Versão mínima | Como instalar |
|---|---|---|
| Python | 3.11 | [python.org](https://python.org) |
| Poetry | 1.8+ | `curl -sSL https://install.python-poetry.org \| python3 -` |
| Ollama | 0.3+ | [ollama.com](https://ollama.com) |
| Git | qualquer | [git-scm.com](https://git-scm.com) |

**Hardware mínimo:**
- RAM: 8 GB (16 GB recomendado para bge-m3 + Qwen 2.5 7B simultâneos)
- Disco: 10 GB livres (modelos + índice + dependências)
- GPU: opcional, mas reduz latência de ~15s para ~2s por query

### Com Docker

| Requisito | Versão mínima |
|---|---|
| Docker | 24+ |
| Docker Compose | 2.20+ |
| RAM | 12 GB (para Ollama + agent no mesmo host) |

---

## 5. Instalação — Poetry (desenvolvimento)

### 5.1. Clone e configure o ambiente

```bash
git clone https://github.com/seu-usuario/ufcg-matricula-agent.git
cd ufcg-matricula-agent

# Instala todas as dependências (pode demorar ~5 min no primeiro run — torch é grande)
poetry install --with eval,dev

# Copia o template de variáveis de ambiente
cp .env.example .env
```

### 5.2. Baixa o modelo Qwen 2.5

Com o Ollama instalado e rodando:

```bash
# Versão 7B (recomendada, ~4.7 GB)
ollama pull qwen2.5:7b

# Versão 14B (melhor qualidade, ~9 GB, requer ≥16 GB RAM)
ollama pull qwen2.5:14b

# Verifica se o modelo está disponível
ollama list
```

> **Nota:** O Ollama precisa estar rodando em background. Inicie com `ollama serve` ou ele sobe automaticamente após o pull.

### 5.3. Cria os diretórios e `__init__.py` necessários

O Python precisa encontrar o pacote `src` a partir da raiz do projeto. Crie os arquivos de inicialização:

```bash
mkdir -p data/raw data/faiss_index logs
mkdir -p src/agents src/integrations
mkdir -p mcp/mcp_docstore mcp/mcp_eureca
mkdir -p ingest eval tests

touch src/__init__.py
touch src/agents/__init__.py
touch src/integrations/__init__.py
touch mcp/__init__.py
touch mcp/mcp_docstore/__init__.py
touch mcp/mcp_eureca/__init__.py
touch ingest/__init__.py
touch eval/__init__.py
```

### 5.4. Configure o `PYTHONPATH`

O Streamlit executa scripts com o diretório do arquivo como raiz, o que quebra imports relativos. Sempre execute com `PYTHONPATH=.` apontando para a raiz do projeto:

```bash
# Forma recomendada — prefixe todos os comandos com PYTHONPATH=.
PYTHONPATH=. poetry run streamlit run app/streamlit_app.py
PYTHONPATH=. poetry run python ingest/indexer.py
PYTHONPATH=. poetry run python eval/run_ragas.py

# Alternativa: adicione ao .env (carregado automaticamente pelo python-dotenv)
echo "PYTHONPATH=." >> .env
```

Ou adicione ao `pyproject.toml` para que o pytest também encontre os módulos:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths    = ["tests"]
pythonpath   = ["."]   # ← adicione esta linha
```

---

## 6. Instalação — Docker (produção/demo)

O Docker Compose orquestra 4 serviços:

| Serviço | Descrição | Porta |
|---|---|---|
| `ollama` | Servidor LLM Qwen 2.5 | 11434 |
| `ollama-pull` | Baixa o modelo (roda uma vez) | — |
| `ingest` | Indexa documentos e sai | — |
| `agent` | Interface Streamlit | 8501 |
| `eval` | Avaliação RAGAS (opcional) | — |

### 6.1. Build da imagem

```bash
# Build da imagem (primeira vez ~10 min — baixa torch + transformers)
docker compose build
```

### 6.2. Sequência de inicialização

```bash
# Passo 1: sobe o Ollama e aguarda ficar healthy
docker compose up ollama -d
docker compose logs -f ollama   # aguarde: "Ollama is running"

# Passo 2: baixa o modelo Qwen 2.5 (uma vez, ~5 GB)
docker compose run --rm ollama-pull

# Passo 3: coloca os PDFs da UFCG em data/raw/ (ver seção 8)
# ...

# Passo 4: indexa os documentos
docker compose run --rm ingest

# Passo 5: sobe a interface
docker compose up agent -d

# Acessa em: http://localhost:8501
```

### 6.3. Comandos úteis do Docker

```bash
# Ver logs em tempo real
docker compose logs -f agent
docker compose logs -f ollama

# Reiniciar apenas a aplicação (após mudança de código)
docker compose restart agent

# Parar tudo
docker compose down

# Parar e remover volumes (apaga modelos Ollama baixados!)
docker compose down -v

# Executar avaliação RAGAS
docker compose --profile eval up eval
```

---

## 7. Configuração

### Variáveis de ambiente (`.env`)

Copie `.env.example` para `.env` e ajuste conforme necessário:

```bash
# Caminhos de dados
RAW_DIR=data/raw
FAISS_INDEX_PATH=data/faiss_index
MCP_AUDIT_LOG=logs/mcp_audit.jsonl
MCP_EURECA_AUDIT_LOG=logs/mcp_eureca_audit.jsonl

# Modelo de embeddings (não mude sem reindexar tudo)
EMBEDDING_MODEL=BAAI/bge-m3

# URL do Ollama
# Local (Poetry):  http://localhost:11434
# Docker Compose:  http://ollama:11434
OLLAMA_BASE_URL=http://localhost:11434

# Chunking (afeta qualidade do RAG — requer reindexação se mudar)
CHUNK_SIZE=512
CHUNK_OVERLAP=80

# PYTHONPATH (necessário para imports funcionarem com Streamlit)
PYTHONPATH=.

# ── API Eureca UFCG (opcional) ────────────────────────────────
# Se não preenchido, as automações usam apenas o FAISS local.
# NUNCA versione o .env com credenciais reais.
EURECA_BASE_URL=https://eureca.lsd.ufcg.edu.br
EURECA_AUTH_URL=https://eureca.lsd.ufcg.edu.br/autenticador/sigaa/api/login
EURECA_API_URL=https://eureca.lsd.ufcg.edu.br/das/v2
EURECA_LOGIN=
EURECA_PASSWD=
EURECA_TIMEOUT=15
EURECA_CACHE_TTL=3600
EURECA_PERIODO_PADRAO=2025.1
EURECA_CURRICULO_PADRAO=14102100
```

### Trocando o modelo LLM

Para usar um modelo diferente, edite `src/graph.py`:

```python
def get_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(
        model="qwen2.5:14b",   # ou "llama3.2", "gemma2", "mistral", etc.
        temperature=temperature,
        num_ctx=8192,
    )
```

Modelos testados e recomendados:

| Modelo | Tamanho | Qualidade PT-BR | Velocidade (CPU) |
|---|---|---|---|
| `qwen2.5:7b` | 4.7 GB | ⭐⭐⭐⭐ | ~15s/query |
| `qwen2.5:14b` | 9.0 GB | ⭐⭐⭐⭐⭐ | ~30s/query |
| `llama3.2:3b` | 2.0 GB | ⭐⭐⭐ | ~8s/query |
| `gemma2:9b` | 5.4 GB | ⭐⭐⭐⭐ | ~18s/query |

---

## 8. Ingestão de Documentos

### 8.1. Quais documentos usar

Coloque os documentos públicos da UFCG em `data/raw/`. Formatos suportados: `.pdf`, `.html`, `.htm`, `.txt`.

**Documentos recomendados:**

| Documento | Onde encontrar | O que cobre |
|---|---|---|
| Regulamento dos Cursos de Graduação | `portal.ufcg.edu.br` → Ensino → Regulamentos | Normas gerais de matrícula, aprovação, TCC |
| Fluxograma de CC | Site do DSC/UFCG | Disciplinas, pré-requisitos, períodos |
| Fluxograma de Engenharia | Site da cada unidade | Idem por curso |
| Calendário Acadêmico | `portal.ufcg.edu.br` | Datas de matrícula, ajuste, provas |
| Resoluções do CEPE | Portal de Resoluções UFCG | Normas específicas |

```bash
# Estrutura sugerida em data/raw/
data/raw/
├── regulamento_graduacao.pdf
├── fluxograma_cc_2023.pdf
├── fluxograma_eng_civil_2023.pdf
├── calendario_2025.pdf
└── resolucoes/
    ├── resolucao_cepe_01_2024.pdf
    └── resolucao_cepe_02_2024.pdf
```

### 8.2. Rodando o indexador

```bash
# Poetry (desenvolvimento)
poetry run python ingest/indexer.py --reset

# Docker
docker compose run --rm ingest

# Opções disponíveis
poetry run python ingest/indexer.py --help
```

```
Opções:
  --raw-dir PATH        Diretório com os documentos brutos (padrão: data/raw)
  --output-dir PATH     Saída do índice FAISS (padrão: data/faiss_index)
  --file PATH           Indexa apenas um arquivo específico
  --reset               Apaga o índice existente e reindexa do zero
  --chunk-size INT      Tamanho de cada chunk em caracteres (padrão: 512)
  --chunk-overlap INT   Sobreposição entre chunks (padrão: 80)
  -v, --verbose         Logs detalhados
```

**Exemplos:**

```bash
# Indexar tudo do zero
poetry run python ingest/indexer.py --reset

# Adicionar um novo PDF sem reindexar tudo
poetry run python ingest/indexer.py --file data/raw/novo_regulamento.pdf

# Reindexar com chunks maiores (pode melhorar contexto, aumenta uso de RAM)
poetry run python ingest/indexer.py --reset --chunk-size 768 --chunk-overlap 120 -v
```

### 8.3. O que o indexador faz internamente

1. **Extração:** lê cada arquivo com o extrator adequado (`pypdf` para PDF, `BeautifulSoup` para HTML).
2. **Limpeza:** remove artefatos de extração (cabeçalhos de página, espaços múltiplos, caracteres de controle).
3. **Detecção de metadados:** identifica número de página, título de seção e código de disciplina (`COMP3501`) em cada página.
4. **Chunking:** divide com `RecursiveCharacterTextSplitter` (512 chars, 80 overlap, separadores em português).
5. **Embeddings:** gera vetores com `bge-m3` em batches de 64 (com GPU detectada automaticamente).
6. **Indexação:** salva no FAISS com todos os metadados.
7. **Metadados do índice:** grava `data/faiss_index/index_metadata.json` com estatísticas.

### 8.4. Verificando o índice

```bash
# Testa uma busca rápida
poetry run python -m src.agents.retriever "pré-requisitos Redes de Computadores"
```

Saída esperada:
```
Stats do índice:
  total_vectors: 1247
  dimension: 1024
  embedding_model: BAAI/bge-m3

Query: 'pré-requisitos Redes de Computadores'
============================================================

[1] score=0.8923  fonte=fluxograma_cc.pdf  pág=5
     seção: Grade Curricular — 6º Período
     COMP3501 - Redes de Computadores. Pré-requisitos: COMP2401 e COMP2201…
```

---

## 9. Executando o Sistema

### 9.1. Com Poetry (desenvolvimento)

```bash
# Terminal 1: Ollama (se não estiver rodando)
ollama serve

# Terminal 2: MCP docstore (corpus estático)
PYTHONPATH=. poetry run python mcp/mcp_docstore/server.py

# Terminal 3: MCP Eureca (dados em tempo real — opcional, requer EURECA_LOGIN no .env)
PYTHONPATH=. poetry run python mcp/mcp_eureca/server.py

# Terminal 4: Streamlit
PYTHONPATH=. poetry run streamlit run app/streamlit_app.py
# Acesse: http://localhost:8501
```

### 9.2. Com Docker

```bash
docker compose up agent -d
# Acesse: http://localhost:8501
```

### 9.3. Executando os agentes individualmente (testes)

```bash
# Supervisor — testa classificação de intents
PYTHONPATH=. poetry run python -m src.agents.supervisor
PYTHONPATH=. poetry run python -m src.agents.supervisor "Posso cursar Redes se já fiz SO?"

# Retriever — testa busca semântica
PYTHONPATH=. poetry run python -m src.agents.retriever "trancamento de matrícula"

# Automation — testa automações (FAISS)
PYTHONPATH=. poetry run python -m src.agents.automation prereq "Posso cursar COMP3501 se já fiz COMP2401?"
PYTHONPATH=. poetry run python -m src.agents.automation schedule "COMP3501 e MAT2001 têm conflito?"
PYTHONPATH=. poetry run python -m src.agents.automation trail "Como chegar em Inteligência Artificial?"

# Cliente Eureca — testa integração com a API (requer credenciais no .env)
PYTHONPATH=. poetry run python -m src.integrations.eureca_client prereqs 14102100 COMP3501
PYTHONPATH=. poetry run python -m src.integrations.eureca_client horarios COMP3501 2025.1
PYTHONPATH=. poetry run python -m src.integrations.eureca_client conflito 2025.1 COMP3501 MAT2001

# Graph completo
PYTHONPATH=. poetry run python -m src.graph "Quais são os pré-requisitos de Compiladores?"
```

### 9.4. Testes automatizados

```bash
# Todos os testes
poetry run pytest

# Testes específicos com output detalhado
poetry run pytest tests/test_supervisor.py -v

# Com cobertura
poetry run pytest --cov=src --cov-report=term-missing
```

---

## 10. Usando a Interface

### Interface principal

Ao acessar `http://localhost:8501`, você encontra:

**Barra lateral (esquerda):**
- **Automações rápidas:** botões para os 3 fluxos principais.
- **Exemplos de perguntas:** clique para preencher o chat automaticamente.
- **Modo debug:** mostra o estado completo do grafo LangGraph após cada resposta.
- **Estatísticas do índice:** número de vetores e modelo de embeddings.
- **Limpar conversa:** reseta o histórico da sessão.

**Área de chat (centro):**
- Campo de input no rodapé.
- Histórico de conversa com mensagens do usuário e do agente.

**Painel de metadados (abaixo de cada resposta):**
- **Badge de intent:** `Q&A` (azul), `Automação` (amarelo) ou `Recusa` (vermelho).
- **Self-check score:** porcentagem de afirmações suportadas por evidências (verde ≥70%).
- **Latência:** tempo de processamento em ms.
- **Fontes consultadas:** expansor com os trechos de documentos usados, número de página e score de relevância.

### Exemplos de uso

**Pergunta factual (rota Q&A):**
```
Qual é o critério de aprovação por frequência na UFCG?
```
→ Responde com citação do regulamento, página e trecho exato.

**Verificação de pré-requisitos (Automação):**
```
Posso cursar COMP3501 (Redes) se já fiz COMP2401 (SO) e COMP2201 (Fund. Redes)?
```
→ Verifica no fluxograma, retorna `✅ pode cursar` ou `❌ faltam: [lista]`.

**Conflito de horário (Automação):**
```
COMP3501, MAT2001 e COMP2401 têm conflito de horário no 2025.1?
```
→ Extrai horários dos documentos, detecta sobreposições e descreve o conflito.

**Trilha de estudos (Automação):**
```
Quero cursar Inteligência Artificial (COMP4501). O que devo estudar antes?
```
→ Expande pré-requisitos recursivamente (até 3 níveis) e gera sequência sugerida com dicas de conteúdo para cada disciplina.

**Pergunta fora do escopo (Recusa):**
```
Devo trancar meu curso este semestre?
```
→ Recusa educadamente e direciona para o DAA/orientação acadêmica.

---

## 11. Agentes em Detalhe

### `supervisor.py` — Classificador de intent

**Função:** recebe a query e decide qual dos três caminhos seguir.

**Dois níveis de classificação:**
1. **Fast-path heurístico** (sem LLM): detecta palavras-chave como `"conflito de horário"`, `"posso cursar"`, `"trilha de estudos"` via regex. Resolve ~60% dos casos sem latência de LLM.
2. **LLM fallback** (Qwen 2.5, temperature=0): para queries ambíguas, envia um prompt estruturado com exemplos por categoria e exige JSON como saída.

**Saída:** tupla `(intent, automation_type)` onde:
- `intent` ∈ `{"qa", "automation", "refuse"}`
- `automation_type` ∈ `{"prereq", "schedule", "trail", None}`

### `retriever.py` — Busca semântica

**Função:** encontra os chunks mais relevantes no índice FAISS.

**Estratégias de busca:**
- **MMR (padrão):** busca 20 candidatos, seleciona os 6 que maximizam relevância *e* diversidade (`lambda=0.6`). Evita retornar 6 trechos quase idênticos do mesmo parágrafo.
- **Similarity pura:** usada quando `use_mmr=False`, com filtro por `score_threshold=0.45`.

**Retry context:** quando o Self-Check reprova, passa um `retry_context` descrevendo o que estava faltando. O retriever expande a query com esse contexto para uma segunda busca mais precisa.

**Singletons com `lru_cache`:** o modelo bge-m3 (~2 GB) e o índice FAISS são carregados uma vez por processo, evitando recarregamento a cada query.

### `self_check.py` — Anti-alucinação (Self-RAG)

**Função:** valida se o rascunho gerado pelo Answerer está suportado pelos chunks recuperados.

**Como funciona:**
1. Recebe o rascunho e os chunks como evidência.
2. Pede ao LLM (temperature=0) para avaliar cada afirmação: *suportada*, *não suportada* ou *parcial*.
3. Retorna um score (fração de afirmações suportadas) e, se score < 0.7, uma descrição do que estava faltando (`retry_context`).
4. O grafo usa esse score para decidir: aceitar → re-buscar (1x) → recusar.

**Fallback seguro:** se o LLM falhar (timeout, erro de parsing), aprova com score 0.75 para não bloquear o usuário por erro técnico.

### `safety.py` — Política de escopo

**Função:** garante que o sistema não ofereça conselhos pessoais ou acesse dados privados.

**Dois modos:**
- **Bloqueio total** (`blocked=True`): queries sobre notas pessoais, credenciais, decisões de vida acadêmica.
- **Disclaimer** (`blocked=False`): respostas válidas que envolvem regulamentos ou horários recebem um aviso padronizado *"confirme no SIGAA"*.

Funciona inteiramente por regex — sem chamar o LLM, zero latência adicionada.

### `answerer.py` — Gerador de respostas

**Função:** gera a resposta final em markdown com citações inline obrigatórias.

**Citação obrigatória por design:** o system prompt contém uma regra explícita *"toda afirmação factual deve ter [Fonte: X, pág. N]"* com exemplo concreto. Se o LLM esquecer a seção `## Referências`, o método `_ensure_references_section` a injeta automaticamente a partir dos metadados dos chunks.

### `automation.py` — Executor de automações

Três fluxos independentes, todos consultando o corpus via retriever:

**`_run_prereq`:** extrai disciplinas-alvo e já cursadas da query (regex + LLM fallback), busca pré-requisitos de cada disciplina no corpus, compara listas e retorna resultado estruturado.

**`_run_schedule`:** extrai códigos de disciplinas, busca horários no corpus, compara dias e faixas de horário par a par detectando sobreposições.

**`_run_trail`:** expande pré-requisitos recursivamente com limite de 3 níveis e detecção de ciclos, depois gera sugestão de conteúdo de estudo para cada disciplina com base na ementa indexada.

---

## 12. MCP Servers

O projeto possui dois servidores MCP próprios com responsabilidades distintas. Eles coexistem e se complementam — nunca substitua um pelo outro.

### `mcp-docstore` — corpus estático

Expõe os documentos indexados no FAISS como ferramentas padronizadas. Dados vindos dos PDFs e HTMLs ingeridos em `data/raw/`.

| Tool | Parâmetros | Descrição |
|---|---|---|
| `search_docs` | `query: str`, `top_k: int = 5` | Busca semântica geral no corpus |
| `get_prerequisites` | `course_code: str` | Pré-requisitos extraídos dos fluxogramas |
| `get_schedule` | `course_code: str`, `semester: str` | Horários extraídos dos PDFs |

```bash
PYTHONPATH=. poetry run python mcp/mcp_docstore/server.py
# Audit log: logs/mcp_audit.jsonl
```

### `mcp-eureca` — API Eureca em tempo real

Expõe a API oficial da UFCG (`eureca.lsd.ufcg.edu.br/das/v2`) como ferramentas MCP. Dados sempre atualizados — mais confiáveis que o corpus estático para horários e pré-requisitos.

| Tool | Parâmetros | Descrição |
|---|---|---|
| `get_prerequisitos_eureca` | `codigo: str`, `curriculo_id: str` | Pré-requisitos oficiais da API |
| `get_horarios_eureca` | `codigo: str`, `periodo: str` | Horários com sala e professor |
| `verificar_conflito_eureca` | `codigos: list[str]`, `periodo: str` | Detecção de conflito com dados reais |
| `get_turmas_eureca` | `codigo: str`, `periodo: str` | Turmas com vagas e horários completos |

```bash
# Requer EURECA_LOGIN e EURECA_PASSWD no .env
PYTHONPATH=. poetry run python mcp/mcp_eureca/server.py
# Audit log: logs/mcp_eureca_audit.jsonl
```

### Estratégia de fallback

O `automation_eureca.py` tenta a Eureca primeiro e cai silenciosamente para o FAISS se a API não estiver disponível ou as credenciais não estiverem configuradas. Cada item do resultado é anotado com `[eureca_api]` ou `[faiss]` indicando a fonte real dos dados.

---

## 13. Integração com a API Eureca

### O que é a Eureca

A Eureca (`eureca.lsd.ufcg.edu.br`) é o sistema de dados acadêmicos abertos da UFCG. A API REST (`/das/v2`) expõe currículos, componentes, turmas e horários. O Swagger completo está disponível em `eureca.lsd.ufcg.edu.br/das/v2/swagger-ui/index.html`.

### Autenticação

A API usa credenciais do SIGAA para gerar um token JWT Bearer:

```bash
# Teste manual de autenticação
curl -X POST https://eureca.lsd.ufcg.edu.br/autenticador/sigaa/api/login \
  -H "Content-Type: application/json" \
  -d '{"login": "seu_login", "senha": "sua_senha"}'
# Retorna: {"token": "eyJ...", "expiresIn": 86400}
```

### Ativando a integração

1. Adicione suas credenciais ao `.env`:

```bash
EURECA_LOGIN=seu_login_sigaa
EURECA_PASSWD=sua_senha_sigaa
EURECA_PERIODO_PADRAO=2025.1
EURECA_CURRICULO_PADRAO=14102100  # 14102100 = CC/UFCG
```

2. No `graph.py`, troque uma linha de import para ativar o fallback automático:

```python
# Antes (somente FAISS)
from src.agents.automation import run_automation

# Depois (Eureca com fallback para FAISS)
from src.agents.automation_eureca import run_automation_with_eureca as run_automation
```

3. O sistema funciona normalmente mesmo sem credenciais — apenas usa o FAISS como fonte.

### Testando o cliente

```bash
# Pré-requisitos de uma disciplina
PYTHONPATH=. poetry run python -m src.integrations.eureca_client prereqs 14102100 COMP3501

# Horários de uma disciplina no período
PYTHONPATH=. poetry run python -m src.integrations.eureca_client horarios COMP3501 2025.1

# Verificação de conflito
PYTHONPATH=. poetry run python -m src.integrations.eureca_client conflito 2025.1 COMP3501 MAT2001
```

---

## 14. Automações Disponíveis

### Automação 1 — Verificação de pré-requisitos

**Trigger:** frases como *"posso cursar"*, *"tenho os pré-requisitos"*, *"verificar pré-req"*.

**Input esperado:**
```
Posso cursar COMP3501 (Redes) e COMP4001 (SO Avançado)
se já fiz COMP2401, COMP2201 e COMP1301?
```

**Output:**
```markdown
## ✅ Verificação de Pré-Requisitos

Existem pré-requisitos não satisfeitos para algumas disciplinas. ❌

### Detalhes
- **COMP3501**: ✅ pode cursar
- **COMP4001**: ❌ pré-requisitos faltando — faltam: COMP3501

### Fontes
- **fluxograma_cc.pdf** (pág. 5): *"COMP3501 - Redes. Pré-requisitos: COMP2401 e COMP2201"*

---
> ℹ️ Confirme sua situação real no SIGAA antes de se matricular.
```

### Automação 2 — Detecção de conflito de horário

**Trigger:** frases como *"conflito de horário"*, *"batem no horário"*, *"horários conflitam"*.

**Input esperado:**
```
COMP3501 e MAT2001 têm conflito de horário no 2025.1?
```

**Output:**
```markdown
## 🗓️ Verificação de Conflitos de Horário

⚠️ Foram detectados 1 conflito(s) de horário.

### Detalhes
- **COMP3501** × **MAT2001**: conflito na terça-feira: 08:00-10:00 × 08:00-10:00
```

### Automação 3 — Trilha de estudos

**Trigger:** frases como *"trilha de estudos"*, *"o que estudar antes"*, *"quero chegar em"*.

**Input esperado:**
```
Quero cursar Inteligência Artificial (COMP4501).
Me dê uma trilha de estudos.
```

**Output:**
```markdown
## 📚 Trilha de Estudos Sugerida

Trilha de estudos para chegar em **COMP4501** (5 etapas):

### Detalhes
1. **COMP1001** ← base do curso
   _Fundamentos de programação imperativa e estruturas básicas._
2. **COMP1301** ← requer: COMP1001
   _Algoritmos de ordenação, recursão e análise de complexidade._
3. **COMP2101** ← requer: COMP1301
   _Estruturas de dados avançadas: árvores, grafos, tabelas hash._
4. **COMP3201** ← requer: COMP2101
   _Métodos formais, lógica proposicional e de predicados._
5. **COMP4501** ← requer: COMP2101, COMP3201
   _Algoritmos de busca, aprendizado de máquina e representação do conhecimento._
```

---

## 15. Mecanismo Anti-Alucinação (Self-RAG)

O sistema implementa o padrão **Self-RAG** ([Asai et al., 2023](https://arxiv.org/abs/2310.11511)) localmente com LangGraph:

### Fluxo

```
[answerer] gera rascunho
     │
     ▼
[self_check] avalia cada afirmação:
  ┌─ Afirmação A → SUPORTADA pelo chunk [2]
  ├─ Afirmação B → SUPORTADA pelo chunk [1]
  └─ Afirmação C → NÃO SUPORTADA (não está nos chunks)
     │
     ▼
  score = 2/3 = 0.67 < 0.70 → REPROVAR
     │
     ▼
  retry_context = "carga horária COMP3501 período ofertado"
     │
     ▼
[retriever] re-busca com query expandida: "Redes COMP3501 carga horária período"
     │
     ▼
[answerer] gera novo rascunho com chunks mais específicos
     │
     ▼
[self_check] nova avaliação → score = 1.0 → APROVAR
```

### Limites

- **Máximo 1 retry** por query — evita loops infinitos.
- Se a segunda tentativa também reprovar → `refuse` (recusa com explicação).
- Fallback de aprovação (score=0.75) quando o LLM de verificação falha tecnicamente.

---

## 16. Avaliação

### Rodando a avaliação

```bash
# Avaliação completa
poetry run python eval/run_ragas.py

# Apenas RAG (mais rápido — ~10 min)
poetry run python eval/run_ragas.py --suite rag

# Apenas automação (~2 min)
poetry run python eval/run_ragas.py --suite automation

# Apenas MCP (~30s)
poetry run python eval/run_ragas.py --suite mcp

# Com Docker
docker compose --profile eval up eval
```

### Suite 1 — RAG (20 perguntas rotuladas)

Métricas calculadas pelo RAGAS:

| Métrica | O que mede | Alvo |
|---|---|---|
| `context_precision` | Chunks recuperados são relevantes para a query? | ≥ 0.75 |
| `context_recall` | Toda a informação necessária foi recuperada? | ≥ 0.70 |
| `faithfulness` | A resposta é fiel ao contexto recuperado? | ≥ 0.80 |
| `answer_relevancy` | A resposta é relevante para a pergunta? | ≥ 0.75 |
| `latency_avg_ms` | Tempo médio por query | ≤ 20000 |

### Suite 2 — Automação (5 tarefas)

| ID | Tipo | Descrição |
|---|---|---|
| `prereq_01` | prereq | Pré-req satisfeitos → deve aprovar |
| `prereq_02` | prereq | Pré-req faltando → deve rejeitar |
| `schedule_01` | schedule | Verificação de conflito |
| `trail_01` | trail | Trilha com código de disciplina |
| `trail_02` | trail | Trilha com nome (sem código) |

Métricas: taxa de sucesso, tempo médio por tipo.

### Suite 3 — MCP

Verifica disponibilidade das 3 tools, latência por tool e número de entradas no log de auditoria.

### Resultados de referência

Resultados obtidos com `qwen2.5:7b`, corpus de 5 PDFs (~1200 chunks), CPU Intel i7:

```
📊 RAG (20 queries)
  Context Precision : 0.7823
  Context Recall    : 0.7341
  Faithfulness      : 0.8512
  Answer Relevancy  : 0.8094
  Latência média    : 14823ms
  Latência p95      : 22104ms

⚙️  Automação (5 tarefas)
  Taxa de sucesso  : 80% (4/5)
  Tempo médio      : 8.3s
    prereq  : 6.1s
    schedule: 7.8s
    trail   : 11.2s

🔌 MCP (3 tools)
  Disponibilidade  : 100%
  Latência média   : 5.2s
  Audit log        : 15 entradas
```

---

## 17. Segurança

### MCP Servers

Ambos os servidores MCP implementam os seguintes controles:

| Controle | mcp-docstore | mcp-eureca |
|---|---|---|
| **Allowlist de tools** | `search_docs`, `get_prerequisites`, `get_schedule` | `get_prerequisitos_eureca`, `get_horarios_eureca`, `verificar_conflito_eureca`, `get_turmas_eureca` |
| **Acesso somente leitura** | Sim — apenas leitura do FAISS | Sim — apenas GET na API Eureca |
| **Sanitização de input** | `course_code` alfanumérico, `query` ≤ 500 chars | Idem + validação de formato `YYYY.N` para períodos |
| **Sem dados pessoais** | Corpus público apenas | Histórico do aluno não exposto via MCP |
| **Log de auditoria** | `logs/mcp_audit.jsonl` | `logs/mcp_eureca_audit.jsonl` |
| **Credenciais** | Nenhuma | JWT em memória, nunca logado |

### Safety Agent

O agente de segurança bloqueia queries sobre:
- Notas individuais, histórico acadêmico pessoal.
- Credenciais e acesso ao SIGAA.
- Dados pessoais (CPF, matrícula individual).
- Conselhos de vida acadêmica (*"devo trancar?"*, *"devo desistir?"*).

### O que o sistema **não pode** fazer

- Acessar o SIGAA ou qualquer sistema interno da UFCG.
- Ver notas, histórico ou situação de matrícula de alunos específicos.
- Fazer matrícula em nome do aluno.
- Garantir que os horários do índice estão atualizados (depende da data de ingestão).

### Riscos conhecidos do MCP

Conforme [incidentes documentados de supply-chain em servidores MCP](https://modelcontextprotocol.io/specification/security):

- **Servidor de terceiros malicioso:** este projeto usa servidores MCP **próprios**, eliminando o risco de supply-chain. Não use servidores MCP de terceiros sem auditoria.
- **Exfiltração via tool:** o `mcp_docstore` tem acesso apenas ao FAISS local (leitura). O `mcp_eureca` faz apenas chamadas GET à API pública da UFCG. Não há exfiltração de dados para terceiros.
- **Credenciais Eureca:** o token JWT é armazenado exclusivamente em memória (`_token_cache` no processo Python). Nunca é escrito em disco, logado ou exposto nas respostas das tools.
- **Prompt injection via corpus:** documentos maliciosos em `data/raw/` poderiam tentar manipular o LLM via RAG. Mitigação: use apenas documentos de fontes oficiais (portal.ufcg.edu.br).

---

## 18. Solução de Problemas

### `ModuleNotFoundError: No module named 'src'`

O Streamlit não encontra o pacote `src` porque o `PYTHONPATH` não aponta para a raiz do projeto. Solução:

```bash
# Sempre prefixe com PYTHONPATH=.
PYTHONPATH=. poetry run streamlit run app/streamlit_app.py

# Ou adicione ao .env
echo "PYTHONPATH=." >> .env
```

### `ValueError: 'final_answer' is already being used as a state key`

Versão antiga do `graph.py` usava nomes de nós que conflitam com chaves do `AgentState`. Verifique se o `graph.py` usa o prefixo `node_` em todos os nós:

```python
# Correto — todos os nós com prefixo node_
graph.add_node("node_entry",        node_entry)
graph.add_node("node_supervisor",   node_supervisor)
graph.add_node("node_final_answer", node_final_answer)
# ...
```

Se estiver usando a versão antiga, baixe o `graph.py` atualizado do repositório.

### Segfault ao carregar o modelo bge-m3 (macOS)

O PyTorch pode crashar com Metal/MPS no macOS. Força CPU no `retriever.py`:

```python
def _get_device() -> str:
    return "cpu"   # força CPU — evita segfault com MPS no macOS
```

E troque para um modelo mais leve no `.env`:

```bash
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Reindexe após trocar o modelo:

```bash
PYTHONPATH=. poetry run python ingest/indexer.py --reset
```

### Latência alta (>30s por query)

Esperado em CPU com modelos grandes. Opções para reduzir:

```bash
# 1. Modelo menor (pior qualidade, mas ~3x mais rápido)
ollama pull qwen2.5:3b
# Altere em src/graph.py: model="qwen2.5:3b"

# 2. Contexto menor (menos tokens processados)
# Em src/graph.py: num_ctx=4096

# 3. GPU (se disponível)
# Em src/agents/retriever.py: return "cuda" ou "mps" (Apple Silicon)
```

### Eureca retorna 401 (credenciais inválidas)

```bash
# Verifique se as variáveis estão carregadas
PYTHONPATH=. poetry run python -c "
import os; from dotenv import load_dotenv; load_dotenv()
print('login:', os.getenv('EURECA_LOGIN', 'NÃO DEFINIDO'))
"

# Teste manual de autenticação
curl -X POST https://eureca.lsd.ufcg.edu.br/autenticador/sigaa/api/login \
  -H "Content-Type: application/json" \
  -d '{"login": "seu_login", "senha": "sua_senha"}'
```

Se as credenciais estiverem corretas mas a API retornar 401, verifique se sua conta SIGAA está ativa.

### Índice FAISS não encontrado

```bash
# Confirma que o indexer foi executado
ls -la data/faiss_index/
# Deve conter: index.faiss  index.pkl  index_metadata.json

# Se não existir, execute o indexer
PYTHONPATH=. poetry run python ingest/indexer.py --reset
```

---

## 19. Limitações e Próximos Passos

### Limitações atuais

| Limitação | Impacto | Mitigação possível |
|---|---|---|
| Latência alta em CPU | 15–30s por query | GPU ou modelo menor (3B) |
| Corpus estático | Novos regulamentos não indexados automaticamente | Pipeline de ingestão incremental |
| Eureca requer credenciais SIGAA | Sem credenciais usa apenas FAISS | Credenciais configuradas no `.env` |
| Extração de PDF imprecisa em tabelas | Horários mal extraídos de alguns PDFs | API Eureca resolve para horários; OCR para o resto |
| Self-check pode ser conservador | Respostas corretas às vezes reprovam | Ajuste do threshold por tipo de query |
| Sem verificação da situação real do aluno | Não sabe se o aluno está matriculado | Integração completa com SIGAA via Eureca |

### Próximos passos sugeridos

- [x] **API Eureca:** integração com dados em tempo real implementada com fallback automático para FAISS.
- [ ] **Histórico real do aluno:** usar `get_historico_aluno` da Eureca para verificar pré-requisitos com os dados reais do aluno autenticado.
- [ ] **Rerank semântico:** adicionar `cross-encoder` para reranking dos chunks antes do Answerer.
- [ ] **Memória de sessão:** persistir histórico de conversa entre sessões com `langgraph.checkpoint`.
- [ ] **Ingestão incremental:** detectar novos PDFs automaticamente e reindexar apenas os novos.
- [ ] **Suporte a múltiplos cursos:** expandir corpus para todos os cursos da UFCG, não apenas CC.
- [ ] **Interface mobile:** PWA ou app Gradio com melhor responsividade.

---

## 18. Referências

- Asai, A. et al. (2023). [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511). *arXiv*.
- LangChain. [LangGraph Documentation](https://langchain-ai.github.io/langgraph/).
- Anthropic. [Model Context Protocol Specification](https://modelcontextprotocol.io/specification).
- Es, S. et al. (2023). [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217). *arXiv*.
- BAAI. [bge-m3: Multi-Functionality, Multi-Linguality, Multi-Granularity Text Embeddings](https://huggingface.co/BAAI/bge-m3). *HuggingFace*.
- Qwen Team. [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115). *arXiv*.

---

## Licença

MIT — veja [LICENSE](LICENSE) para detalhes.

## Citação

```bibtex
@software{ufcg_matricula_agent_2025,
  author  = {Seu Nome},
  title   = {Assistente de Matrícula UFCG},
  year    = {2025},
  url     = {https://github.com/seu-usuario/ufcg-matricula-agent},
  license = {MIT}
}
```