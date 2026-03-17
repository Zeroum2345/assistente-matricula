# eval/run_ragas.py
# Avaliação do sistema com RAGAS + métricas de automação e MCP
#
# Executa três baterias de avaliação:
#   1. RAG (Q&A):    Context Precision/Recall, Faithfulness, Answer Relevancy, latência
#   2. Automação:    taxa de sucesso, nº médio de steps, tempo médio (5 tarefas)
#   3. MCP:          disponibilidade das tools, latência por tool
#
# Uso:
#   python eval/run_ragas.py                    # avalia tudo
#   python eval/run_ragas.py --suite rag        # só RAG
#   python eval/run_ragas.py --suite automation # só automação
#   python eval/run_ragas.py --suite mcp        # só MCP
#   python eval/run_ragas.py --output eval/results.json

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset de avaliação RAG — 20 perguntas rotuladas
# ---------------------------------------------------------------------------

QA_DATASET: list[dict] = [
    # Pré-requisitos
    {
        "question":         "Quais são os pré-requisitos de Redes de Computadores (COMP3501)?",
        "ground_truth":     "Os pré-requisitos de COMP3501 são Sistemas Operacionais (COMP2401) e Fundamentos de Redes (COMP2201).",
        "expected_sources": ["fluxograma_cc.pdf"],
    },
    {
        "question":         "Quais são os pré-requisitos de Compiladores?",
        "ground_truth":     "Compiladores requer Linguagens de Programação e Estruturas de Dados como pré-requisitos.",
        "expected_sources": ["fluxograma_cc.pdf"],
    },
    {
        "question":         "Quais disciplinas são pré-requisito de Inteligência Artificial?",
        "ground_truth":     "Inteligência Artificial requer Estruturas de Dados e Algoritmos como pré-requisitos diretos.",
        "expected_sources": ["fluxograma_cc.pdf"],
    },
    # Regulamento de matrícula
    {
        "question":         "Como funciona o trancamento de matrícula na UFCG?",
        "ground_truth":     "O aluno pode trancar a matrícula em disciplinas dentro do prazo estabelecido no calendário acadêmico, sem reprovação registrada.",
        "expected_sources": ["regulamento_graduacao.pdf"],
    },
    {
        "question":         "Quantos créditos são necessários para concluir o curso de Ciência da Computação?",
        "ground_truth":     "O curso de Ciência da Computação da UFCG exige o cumprimento de todos os créditos obrigatórios e optativos definidos no currículo.",
        "expected_sources": ["fluxograma_cc.pdf", "regulamento_graduacao.pdf"],
    },
    {
        "question":         "O que é crédito especial na UFCG?",
        "ground_truth":     "Crédito especial é uma modalidade de matrícula em disciplinas fora do currículo regular, sujeita a aprovação do colegiado do curso.",
        "expected_sources": ["regulamento_graduacao.pdf"],
    },
    {
        "question":         "Qual é o prazo máximo para integralização do curso de CC?",
        "ground_truth":     "O prazo máximo de integralização é de 12 semestres para o curso de Ciência da Computação.",
        "expected_sources": ["regulamento_graduacao.pdf"],
    },
    {
        "question":         "Como funciona a segunda chamada de provas na UFCG?",
        "ground_truth":     "O aluno que não comparecer a uma avaliação pode solicitar segunda chamada mediante justificativa dentro do prazo regulamentar.",
        "expected_sources": ["regulamento_graduacao.pdf"],
    },
    # Estrutura curricular
    {
        "question":         "Quais são as disciplinas obrigatórias do primeiro período de CC?",
        "ground_truth":     "No primeiro período de CC constam disciplinas introdutórias de Programação, Matemática Discreta e Cálculo.",
        "expected_sources": ["fluxograma_cc.pdf"],
    },
    {
        "question":         "Qual a carga horária total do curso de Ciência da Computação?",
        "ground_truth":     "A carga horária total do curso de Ciência da Computação é de aproximadamente 3.200 horas.",
        "expected_sources": ["fluxograma_cc.pdf"],
    },
    # Horários e calendário
    {
        "question":         "O que é período de ajuste de matrícula?",
        "ground_truth":     "O período de ajuste é um prazo após a matrícula regular em que o aluno pode incluir ou cancelar disciplinas.",
        "expected_sources": ["regulamento_graduacao.pdf"],
    },
    {
        "question":         "Como é calculada a média final nas disciplinas da UFCG?",
        "ground_truth":     "A média final é calculada com base nas avaliações realizadas durante o semestre, conforme critérios definidos pelo docente e pelo regulamento.",
        "expected_sources": ["regulamento_graduacao.pdf"],
    },
    # Disciplinas optativas e eletivas
    {
        "question":         "Quantos créditos optativos são necessários em CC?",
        "ground_truth":     "O currículo de CC exige um número mínimo de créditos optativos a ser cumprido dentro das trilhas disponíveis.",
        "expected_sources": ["fluxograma_cc.pdf"],
    },
    {
        "question":         "O que é disciplina eletiva e como difere de optativa?",
        "ground_truth":     "Disciplinas eletivas são escolhidas livremente pelo aluno de qualquer área, enquanto optativas pertencem às trilhas definidas pelo colegiado do curso.",
        "expected_sources": ["regulamento_graduacao.pdf"],
    },
    # Reprovação e aprovação
    {
        "question":         "Qual é o critério de aprovação por frequência na UFCG?",
        "ground_truth":     "O aluno deve ter frequência mínima de 75% nas aulas para não ser reprovado por falta.",
        "expected_sources": ["regulamento_graduacao.pdf"],
    },
    {
        "question":         "O que acontece se o aluno reprovar duas vezes na mesma disciplina obrigatória?",
        "ground_truth":     "O regulamento prevê orientação acadêmica obrigatória para alunos que reprovam repetidamente em disciplinas obrigatórias.",
        "expected_sources": ["regulamento_graduacao.pdf"],
    },
    # Equivalências
    {
        "question":         "Como funciona o aproveitamento de estudos de outro curso?",
        "ground_truth":     "O aluno pode solicitar aproveitamento de disciplinas cursadas em outro curso mediante análise do colegiado, que verifica a compatibilidade de conteúdos.",
        "expected_sources": ["regulamento_graduacao.pdf"],
    },
    {
        "question":         "É possível cursar disciplinas de outro curso como optativa?",
        "ground_truth":     "Sim, o aluno pode cursar disciplinas de outros cursos como crédito eletivo, respeitando os limites estabelecidos no currículo.",
        "expected_sources": ["regulamento_graduacao.pdf"],
    },
    # TCC e estágio
    {
        "question":         "Quais são os pré-requisitos para iniciar o TCC em CC?",
        "ground_truth":     "Para iniciar o TCC, o aluno deve ter cumprido um percentual mínimo de créditos obrigatórios, conforme o regulamento do curso.",
        "expected_sources": ["regulamento_graduacao.pdf", "fluxograma_cc.pdf"],
    },
    {
        "question":         "O estágio supervisionado é obrigatório em CC na UFCG?",
        "ground_truth":     "O estágio supervisionado pode ser obrigatório ou optativo dependendo do currículo vigente do curso de Ciência da Computação.",
        "expected_sources": ["fluxograma_cc.pdf"],
    },
]

# ---------------------------------------------------------------------------
# Dataset de automação — 5 tarefas com input/output esperado
# ---------------------------------------------------------------------------

AUTOMATION_TASKS: list[dict] = [
    {
        "id":            "prereq_01",
        "type":          "prereq",
        "query":         "Posso cursar COMP3501 se já fiz COMP2401 e COMP2201?",
        "expected_keys": ["summary", "details", "sources"],
        "expected_outcome": "pode cursar",   # deve estar no summary
        "description":   "Pré-requisitos satisfeitos — deve aprovar",
    },
    {
        "id":            "prereq_02",
        "type":          "prereq",
        "query":         "Posso cursar Compiladores se ainda não fiz Estruturas de Dados?",
        "expected_keys": ["summary", "details", "sources"],
        "expected_outcome": "faltam",        # deve estar no summary ou details
        "description":   "Pré-requisito faltando — deve rejeitar",
    },
    {
        "id":            "schedule_01",
        "type":          "schedule",
        "query":         "COMP3501 e COMP2401 têm conflito de horário no 2025.1?",
        "expected_keys": ["summary", "details", "sources"],
        "expected_outcome": None,            # qualquer resultado válido
        "description":   "Verificação de conflito — qualquer resposta estruturada é válida",
    },
    {
        "id":            "trail_01",
        "type":          "trail",
        "query":         "Quero cursar Inteligência Artificial (COMP4501). Me dê uma trilha de estudos.",
        "expected_keys": ["summary", "details", "sources"],
        "expected_outcome": "trilha",        # deve estar no summary
        "description":   "Geração de trilha — deve retornar sequência de disciplinas",
    },
    {
        "id":            "trail_02",
        "type":          "trail",
        "query":         "O que devo estudar antes de Compiladores?",
        "expected_keys": ["summary", "details", "sources"],
        "expected_outcome": "trilha",
        "description":   "Trilha por nome da disciplina (sem código)",
    },
]

# ---------------------------------------------------------------------------
# Suite 1 — Avaliação RAG com RAGAS
# ---------------------------------------------------------------------------

def run_rag_evaluation(dataset: list[dict]) -> dict:
    """
    Avalia o pipeline RAG com as métricas do RAGAS:
      - context_precision
      - context_recall
      - faithfulness
      - answer_relevancy
    Também mede latência por query.
    """
    logger.info("[eval/rag] iniciando avaliação RAG (%d queries)", len(dataset))

    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset
    except ImportError:
        logger.error("Instale ragas e datasets: pip install ragas datasets")
        return {"error": "ragas não instalado"}

    from src.graph import get_compiled_graph
    from src.graph import AgentState
    from langchain_core.messages import HumanMessage

    graph = get_compiled_graph()

    questions, answers, contexts, ground_truths, latencies = [], [], [], [], []

    for i, item in enumerate(dataset):
        q = item["question"]
        gt = item["ground_truth"]
        logger.info("[eval/rag] (%d/%d) %s", i + 1, len(dataset), q[:60])

        t0 = time.time()
        try:
            initial: AgentState = {
                "messages":          [HumanMessage(content=q)],
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
            result = graph.invoke(initial)
            elapsed_ms = int((time.time() - t0) * 1000)

            answer   = result.get("final_answer", "")
            chunks   = result.get("retrieved_chunks", [])
            ctx_list = [c.get("text", "") for c in chunks]

        except Exception as exc:
            logger.error("[eval/rag] erro na query %d: %s", i + 1, exc)
            answer, ctx_list, elapsed_ms = "", [], 0

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx_list)
        ground_truths.append(gt)
        latencies.append(elapsed_ms)

    # Monta dataset RAGAS
    ragas_dataset = Dataset.from_dict({
        "question":   questions,
        "answer":     answers,
        "contexts":   contexts,
        "ground_truth": ground_truths,
    })

    # ---------------------------------------------------------------------------
    # Configura LLM e embeddings locais para o RAGAS (sem OpenAI)
    # ---------------------------------------------------------------------------
    import os
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ollama_url  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    ragas_llm = LangchainLLMWrapper(
        ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            base_url=ollama_url,
            temperature=0.0,
            num_ctx=4096,
        )
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    # Injeta LLM e embeddings em cada métrica explicitamente
    for metric in [context_precision, context_recall, faithfulness, answer_relevancy]:
        if hasattr(metric, "llm"):
            metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    logger.info("[eval/rag] rodando métricas RAGAS com Ollama (%s) + %s...", ollama_url, embed_model)
    ragas_result = evaluate(
        ragas_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    def _score(key: str) -> float:
        """Extrai média do score — RAGAS pode retornar float ou list[float]."""
        val = ragas_result[key]
        if isinstance(val, list):
            valid = [v for v in val if v is not None and str(v) != "nan"]
            return round(sum(valid) / len(valid), 4) if valid else 0.0
        if val is None or str(val) == "nan":
            return 0.0
        return round(float(val), 4)

    result_dict = {
        "n_queries":         len(dataset),
        "context_precision": _score("context_precision"),
        "context_recall":    _score("context_recall"),
        "faithfulness":      _score("faithfulness"),
        "answer_relevancy":  _score("answer_relevancy"),
        "latency_avg_ms":    round(avg_latency),
        "latency_p95_ms":    round(p95_latency),
        "per_query":         [
            {
                "question":   questions[i][:80],
                "latency_ms": latencies[i],
                "n_contexts": len(contexts[i]),
            }
            for i in range(len(questions))
        ],
    }

    logger.info("[eval/rag] resultados: %s",
                {k: v for k, v in result_dict.items() if k != "per_query"})
    return result_dict


# ---------------------------------------------------------------------------
# Suite 2 — Avaliação de automação
# ---------------------------------------------------------------------------

def run_automation_evaluation(tasks: list[dict]) -> dict:
    """
    Avalia os três tipos de automação:
      - taxa de sucesso (output tem as chaves esperadas e conteúdo correto)
      - número de steps no grafo (aproximado pelo número de nós percorridos)
      - tempo médio de execução por tipo
    """
    logger.info("[eval/automation] iniciando avaliação de automação (%d tarefas)", len(tasks))

    from src.agents.automation import run_automation

    results = []
    success_count = 0
    times_by_type: dict[str, list[float]] = {}

    for task in tasks:
        tid   = task["id"]
        atype = task["type"]
        query = task["query"]
        expected_keys    = task["expected_keys"]
        expected_outcome = task.get("expected_outcome")

        logger.info("[eval/automation] tarefa %s (%s)", tid, atype)
        t0 = time.time()

        try:
            result  = run_automation(atype, query)
            elapsed = time.time() - t0

            # Verifica chaves esperadas
            has_keys = all(k in result for k in expected_keys)

            # Verifica conteúdo esperado no summary + details
            has_outcome = True
            if expected_outcome:
                combined = (result.get("summary", "") + " ".join(result.get("details", []))).lower()
                has_outcome = expected_outcome.lower() in combined

            success = has_keys and has_outcome and result.get("type") != "error"

        except Exception as exc:
            logger.error("[eval/automation] erro na tarefa %s: %s", tid, exc)
            success, elapsed, result = False, time.time() - t0, {}

        if success:
            success_count += 1

        times_by_type.setdefault(atype, []).append(elapsed)

        results.append({
            "id":          tid,
            "type":        atype,
            "success":     success,
            "elapsed_s":   round(elapsed, 2),
            "description": task.get("description", ""),
            "result_type": result.get("type", "error"),
        })

    # Estatísticas
    n = len(tasks)
    avg_time = sum(r["elapsed_s"] for r in results) / n if n else 0
    avg_by_type = {
        t: round(sum(ts) / len(ts), 2)
        for t, ts in times_by_type.items()
    }

    result_dict = {
        "n_tasks":       n,
        "success_count": success_count,
        "success_rate":  round(success_count / n, 4) if n else 0,
        "avg_elapsed_s": round(avg_time, 2),
        "avg_by_type_s": avg_by_type,
        "per_task":      results,
    }

    logger.info("[eval/automation] taxa de sucesso: %d/%d (%.0f%%)",
                success_count, n, result_dict["success_rate"] * 100)
    return result_dict


# ---------------------------------------------------------------------------
# Suite 3 — Avaliação do MCP Server
# ---------------------------------------------------------------------------

def run_mcp_evaluation() -> dict:
    """
    Verifica a disponibilidade e latência das tools do MCP docstore.
    Testa cada tool com uma chamada simples e mede o tempo de resposta.
    """
    logger.info("[eval/mcp] iniciando avaliação do MCP server")

    # Importa as tools diretamente (sem subprocesso MCP para simplicidade do eval)
    try:
        from mcp.mcp_docstore.server import search_docs, get_prerequisites, get_schedule
    except ImportError as exc:
        logger.error("[eval/mcp] não foi possível importar o MCP server: %s", exc)
        logger.error("[eval/mcp] certifique-se de que:")
        logger.error("  1. O índice FAISS existe em data/faiss_index/")
        logger.error("  2. O arquivo mcp/mcp_docstore/server.py existe")
        logger.error("  3. PYTHONPATH=. está configurado")
        return {"error": f"ImportError: {exc}"}
    except Exception as exc:
        logger.error("[eval/mcp] erro ao importar MCP server: %s", exc)
        return {"error": str(exc)}

    tool_tests = [
        {
            "tool":   "search_docs",
            "fn":     lambda: search_docs("pré-requisitos matrícula UFCG", top_k=3),
            "check":  lambda r: isinstance(r, list) and len(r) > 0,
        },
        {
            "tool":   "get_prerequisites",
            "fn":     lambda: get_prerequisites("COMP3501"),
            "check":  lambda r: isinstance(r, dict) and "prerequisites" in r,
        },
        {
            "tool":   "get_schedule",
            "fn":     lambda: get_schedule("COMP3501", "2025.1"),
            "check":  lambda r: isinstance(r, dict) and "schedule_text" in r,
        },
    ]

    tool_results = []
    for test in tool_tests:
        tool_name = test["tool"]
        t0 = time.time()
        try:
            result  = test["fn"]()
            elapsed = time.time() - t0
            ok      = test["check"](result)
            error   = None
        except Exception as exc:
            elapsed = time.time() - t0
            ok      = False
            error   = str(exc)
            result  = None

        logger.info("[eval/mcp] %-20s ok=%s  elapsed=%.2fs", tool_name, ok, elapsed)
        tool_results.append({
            "tool":       tool_name,
            "available":  ok,
            "elapsed_s":  round(elapsed, 3),
            "error":      error,
        })

    available_count = sum(1 for r in tool_results if r["available"])
    avg_latency     = sum(r["elapsed_s"] for r in tool_results) / len(tool_results)

    # Verifica o arquivo de audit log
    from pathlib import Path
    audit_path = Path("logs/mcp_audit.jsonl")
    audit_entries = 0
    if audit_path.exists():
        audit_entries = sum(1 for _ in audit_path.open())

    return {
        "n_tools":          len(tool_results),
        "available_count":  available_count,
        "availability_rate": round(available_count / len(tool_results), 4),
        "avg_latency_s":    round(avg_latency, 3),
        "audit_log_entries": audit_entries,
        "audit_log_path":   str(audit_path),
        "per_tool":         tool_results,
        "security": {
            "allowlist":          ["search_docs", "get_prerequisites", "get_schedule"],
            "write_access":       False,
            "personal_data":      False,
            "audit_logging":      True,
            "input_sanitization": True,
        },
    }


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

def run_all(suite: str | None = None, output_path: Path | None = None) -> dict:
    """
    Executa todas as suites de avaliação (ou apenas a especificada).
    Salva os resultados em JSON se output_path for fornecido.
    """
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suite":        suite or "all",
    }

    if suite in (None, "rag"):
        logger.info("=== Suite: RAG ===")
        report["rag"] = run_rag_evaluation(QA_DATASET)

    if suite in (None, "automation"):
        logger.info("=== Suite: Automação ===")
        report["automation"] = run_automation_evaluation(AUTOMATION_TASKS)

    if suite in (None, "mcp"):
        logger.info("=== Suite: MCP ===")
        report["mcp"] = run_mcp_evaluation()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("Resultados salvos em %s", output_path)

    return report


def _print_summary(report: dict) -> None:
    """Imprime resumo legível no terminal."""
    print("\n" + "=" * 60)
    print("RELATÓRIO DE AVALIAÇÃO — Assistente de Matrícula UFCG")
    print("=" * 60)

    if "rag" in report:
        r = report["rag"]
        if "error" in r:
            print(f"\n📊 RAG — erro: {r['error']}")
        else:
            print(f"\n📊 RAG ({r.get('n_queries', '?')} queries)")
            print(f"  Context Precision : {r.get('context_precision', 'n/a')}")
            print(f"  Context Recall    : {r.get('context_recall', 'n/a')}")
            print(f"  Faithfulness      : {r.get('faithfulness', 'n/a')}")
            print(f"  Answer Relevancy  : {r.get('answer_relevancy', 'n/a')}")
            print(f"  Latência média    : {r.get('latency_avg_ms', 'n/a')}ms")
            print(f"  Latência p95      : {r.get('latency_p95_ms', 'n/a')}ms")

    if "automation" in report:
        r = report["automation"]
        if "error" in r:
            print(f"\n⚙️  Automação — erro: {r['error']}")
        else:
            print(f"\n⚙️  Automação ({r.get('n_tasks', '?')} tarefas)")
            print(f"  Taxa de sucesso  : {r.get('success_rate', 0):.0%} ({r.get('success_count', '?')}/{r.get('n_tasks', '?')})")
            print(f"  Tempo médio      : {r.get('avg_elapsed_s', 'n/a')}s")
            for atype, avg in r.get("avg_by_type_s", {}).items():
                print(f"    {atype:<12}: {avg}s")

    if "mcp" in report:
        r = report["mcp"]
        if "error" in r:
            print(f"\n🔌 MCP — erro: {r['error']}")
            print("   Verifique se o MCP server está rodando e o índice FAISS foi gerado.")
        else:
            print(f"\n🔌 MCP ({r.get('n_tools', '?')} tools)")
            print(f"  Disponibilidade  : {r.get('availability_rate', 0):.0%}")
            print(f"  Latência média   : {r.get('avg_latency_s', 'n/a')}s")
            print(f"  Audit log        : {r.get('audit_log_entries', 0)} entradas")
            for t in r.get("per_tool", []):
                icon = "✅" if t.get("available") else "❌"
                err  = f"  ({t['error']})" if t.get("error") else ""
                print(f"  {icon} {t['tool']:<22} {t.get('elapsed_s', '?')}s{err}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Avaliação do Assistente de Matrícula UFCG")
    parser.add_argument(
        "--suite",
        choices=["rag", "automation", "mcp"],
        default=None,
        help="Suite específica (padrão: todas)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results.json"),
        help="Caminho do arquivo JSON de saída (padrão: eval/results.json)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    report = run_all(suite=args.suite, output_path=args.output)
    _print_summary(report)