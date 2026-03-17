from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv(override=True)

# Caminho do índice FAISS gerado pelo ingest/indexer.py
FAISS_INDEX_PATH: str = os.getenv(
    "FAISS_INDEX_PATH",
    str(Path(__file__).parent.parent.parent / "data" / "faiss_index"),
)

# Modelo de embeddings (multilingual, ótimo para PT-BR)
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
)

# Parâmetros padrão de busca
DEFAULT_TOP_K: int = 6
DEFAULT_SCORE_THRESHOLD: float = 0.45   # distância cosine; abaixo disso, descarta
MMR_LAMBDA: float = 0.6                 # 0=máxima diversidade, 1=máxima relevância
MMR_FETCH_K: int = 20                   # quantos buscar antes do MMR filtrar


# ---------------------------------------------------------------------------
# Estrutura de um chunk recuperado
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """Um trecho de documento recuperado com metadados completos."""
    text: str                          # Conteúdo do chunk
    source: str                        # Nome do arquivo/URL de origem
    page: int | None                   # Página no PDF (None se HTML/web)
    score: float                       # Score de similaridade (0–1, maior = melhor)
    chunk_id: str = ""                 # ID único do chunk no índice
    section: str = ""                  # Seção/título mais próximo (se disponível)
    course_code: str = ""              # Código de disciplina mencionado no chunk
    excerpt: str = field(init=False)   # Trecho curto para citação (primeiros 200 chars)

    def __post_init__(self):
        # Gera excerpt limpo (sem quebras de linha extras)
        clean = re.sub(r"\s+", " ", self.text).strip()
        self.excerpt = clean[:200] + ("…" if len(clean) > 200 else "")

    def to_dict(self) -> dict:
        return {
            "text":        self.text,
            "source":      self.source,
            "page":        self.page,
            "score":       round(self.score, 4),
            "chunk_id":    self.chunk_id,
            "section":     self.section,
            "course_code": self.course_code,
            "excerpt":     self.excerpt,
        }


# ---------------------------------------------------------------------------
# Singleton: carrega embeddings e índice FAISS uma única vez
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_embeddings() -> HuggingFaceEmbeddings:
    """
    Carrega o modelo bge-m3 via HuggingFace.
    O lru_cache garante que o modelo seja carregado uma só vez por processo.
    """
    logger.info("[retriever] carregando embeddings: %s", EMBEDDING_MODEL)
    t0 = time.time()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": _get_device()},
        encode_kwargs={
            "normalize_embeddings": True,   # necessário para cosine similarity
            "batch_size": 32,
        },
    )

    logger.info("[retriever] embeddings carregados em %.1fs", time.time() - t0)
    return embeddings


@lru_cache(maxsize=1)
def _load_vectorstore() -> LangchainFAISS:
    """
    Carrega o índice FAISS do disco.
    Requer que ingest/indexer.py já tenha sido executado.
    """
    index_path = Path(FAISS_INDEX_PATH)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Índice FAISS não encontrado em: {index_path}\n"
            "Execute 'python ingest/indexer.py' para criar o índice."
        )

    logger.info("[retriever] carregando índice FAISS de: %s", index_path)
    t0 = time.time()

    vectorstore = LangchainFAISS.load_local(
        folder_path=str(index_path),
        embeddings=_load_embeddings(),
        allow_dangerous_deserialization=True,  # necessário no LangChain >= 0.2
    )

    n_docs = vectorstore.index.ntotal
    logger.info("[retriever] índice carregado: %d vetores (%.1fs)", n_docs, time.time() - t0)
    return vectorstore


def _get_device() -> str:
    """Detecta se há GPU disponível; usa CPU como fallback."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("[retriever] GPU detectada: %s", torch.cuda.get_device_name(0))
            return "cuda"
    except ImportError:
        pass
    logger.info("[retriever] usando CPU para embeddings")
    return "cpu"


# ---------------------------------------------------------------------------
# Busca principal
# ---------------------------------------------------------------------------

def retrieve_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    use_mmr: bool = True,
    retry_context: str | None = None,
) -> list[dict]:
    """
    Busca semântica no índice FAISS.

    Args:
        query:            Pergunta do usuário (ou query reformulada pelo Self-Check).
        top_k:            Número máximo de chunks a retornar.
        score_threshold:  Descarta chunks com score abaixo deste valor.
        use_mmr:          Se True, aplica MMR para diversidade nos resultados.
        retry_context:    Contexto extra fornecido pelo Self-Check para expandir a query.

    Returns:
        Lista de dicts com os campos de RetrievedChunk.to_dict().
        Retorna lista vazia se o índice não estiver disponível.
    """
    if not query or not query.strip():
        logger.warning("[retriever] query vazia → retornando lista vazia")
        return []

    # Expande a query se vier de um retry
    effective_query = _expand_query(query, retry_context) if retry_context else query
    logger.info("[retriever] query efetiva: %r", effective_query[:120])

    try:
        vectorstore = _load_vectorstore()
    except FileNotFoundError as exc:
        logger.error("[retriever] %s", exc)
        return []

    t0 = time.time()

    if use_mmr:
        raw_docs = _search_mmr(vectorstore, effective_query, top_k)
    else:
        raw_docs = _search_similarity(vectorstore, effective_query, top_k, score_threshold)

    chunks = _parse_docs(raw_docs, score_threshold)

    # Ordena por score decrescente
    chunks.sort(key=lambda c: c.score, reverse=True)

    logger.info(
        "[retriever] %d chunks retornados (threshold=%.2f, mmr=%s, %.2fs)",
        len(chunks), score_threshold, use_mmr, time.time() - t0,
    )

    if not chunks:
        logger.warning("[retriever] nenhum chunk acima do threshold %.2f", score_threshold)

    return [c.to_dict() for c in chunks]


# ---------------------------------------------------------------------------
# Estratégias de busca
# ---------------------------------------------------------------------------

def _search_similarity(
    vectorstore: LangchainFAISS,
    query: str,
    top_k: int,
    score_threshold: float,
) -> list[tuple[Any, float]]:
    """Busca por similaridade cosine pura com score threshold."""
    return vectorstore.similarity_search_with_relevance_scores(
        query,
        k=top_k,
        score_threshold=score_threshold,
    )


def _search_mmr(
    vectorstore: LangchainFAISS,
    query: str,
    top_k: int,
) -> list[tuple[Any, float]]:
    """
    Busca com MMR (Maximal Marginal Relevance).
    Busca MMR_FETCH_K candidatos, depois seleciona top_k balanceando
    relevância e diversidade. Evita chunks muito similares entre si.
    """
    mmr_docs = vectorstore.max_marginal_relevance_search(
        query,
        k=top_k,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA,
    )

    # MMR não retorna scores diretamente; recalcula por similaridade
    if not mmr_docs:
        return []

    embeddings = _load_embeddings()
    query_vec = np.array(embeddings.embed_query(query), dtype=np.float32)

    result = []
    for doc in mmr_docs:
        doc_vec = np.array(
            embeddings.embed_documents([doc.page_content])[0],
            dtype=np.float32,
        )
        # Cosine similarity (vetores já normalizados pelo bge-m3)
        score = float(np.dot(query_vec, doc_vec))
        result.append((doc, score))

    return result


# ---------------------------------------------------------------------------
# Parser: Document LangChain → RetrievedChunk
# ---------------------------------------------------------------------------

def _parse_docs(
    raw_docs: list[tuple[Any, float]],
    score_threshold: float,
) -> list[RetrievedChunk]:
    """
    Converte os Documents do LangChain em RetrievedChunk,
    filtrando pelo threshold e extraindo metadados.

    Metadados esperados no Document (definidos pelo ingest/indexer.py):
      - source:      str  — nome do arquivo ou URL
      - page:        int  — número da página (PDFs)
      - section:     str  — título da seção mais próxima
      - course_code: str  — código de disciplina (ex: COMP1001)
      - chunk_id:    str  — identificador único do chunk
    """
    chunks = []
    for doc, score in raw_docs:
        if score < score_threshold:
            continue

        meta = doc.metadata or {}
        chunk = RetrievedChunk(
            text=doc.page_content,
            source=meta.get("source", "desconhecido"),
            page=meta.get("page"),
            score=score,
            chunk_id=meta.get("chunk_id", ""),
            section=meta.get("section", ""),
            course_code=meta.get("course_code", ""),
        )
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Expansão de query para retry (Self-Check)
# ---------------------------------------------------------------------------

def _expand_query(original_query: str, retry_context: str) -> str:
    """
    Expande a query original com contexto adicional fornecido pelo Self-Check.
    O Self-Check indica quais informações estavam faltando; o retriever
    usa isso para refinar a busca.

    Exemplo:
        original_query = "pré-requisitos de Redes"
        retry_context  = "informações sobre COMP1001 e carga horária"
        → "pré-requisitos de Redes COMP1001 carga horária"
    """
    combined = f"{original_query} {retry_context}"
    # Remove duplicatas de palavras mantendo a ordem
    seen = set()
    words = []
    for word in combined.split():
        norm = word.lower().strip(".,;:")
        if norm not in seen and len(norm) > 2:
            seen.add(norm)
            words.append(word)
    expanded = " ".join(words)
    logger.debug("[retriever] query expandida: %r → %r", original_query, expanded)
    return expanded


# ---------------------------------------------------------------------------
# Utilitários de inspeção (úteis para debugging e avaliação RAGAS)
# ---------------------------------------------------------------------------

def get_index_stats() -> dict:
    """Retorna estatísticas do índice FAISS carregado."""
    try:
        vs = _load_vectorstore()
        return {
            "total_vectors": vs.index.ntotal,
            "dimension":     vs.index.d,
            "index_path":    FAISS_INDEX_PATH,
            "embedding_model": EMBEDDING_MODEL,
        }
    except FileNotFoundError as exc:
        return {"error": str(exc)}


def retrieve_by_course_code(course_code: str, top_k: int = 4) -> list[dict]:
    """
    Busca chunks que mencionam um código de disciplina específico.
    Útil para o Automation Agent ao verificar pré-requisitos.
    """
    query = f"disciplina {course_code} pré-requisito ementa carga horária"
    return retrieve_chunks(query, top_k=top_k, use_mmr=False)


def format_chunks_for_prompt(chunks: list[dict], max_chars: int = 3000) -> str:
    """
    Formata os chunks recuperados para incluir no prompt do LLM.
    Respeita um limite de caracteres para não estourar o contexto.

    Formato de saída:
        [1] Fonte: regulamento_graduacao.pdf, pág. 12, seção: "Matrícula"
        "texto do chunk..."

        [2] Fonte: fluxograma_cc.pdf, pág. 3
        "texto do chunk..."
    """
    lines = []
    total = 0

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "?")
        page = chunk.get("page")
        section = chunk.get("section", "")

        header_parts = [f"Fonte: {source}"]
        if page:
            header_parts.append(f"pág. {page}")
        if section:
            header_parts.append(f'seção: "{section}"')

        header = f"[{i}] " + ", ".join(header_parts)
        body = f'"{chunk.get("text", "")}"'

        entry = f"{header}\n{body}\n"
        entry_len = len(entry)

        if total + entry_len > max_chars:
            lines.append(f"[... {len(chunks) - i + 1} chunks omitidos por limite de contexto]")
            break

        lines.append(entry)
        total += entry_len

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI para testes rápidos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "pré-requisitos de Redes de Computadores"

    print(f"\nStats do índice:")
    stats = get_index_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\nQuery: {query!r}")
    print("=" * 60)

    chunks = retrieve_chunks(query, top_k=5)

    if not chunks:
        print("Nenhum chunk recuperado. Verifique se o índice foi gerado.")
        sys.exit(1)

    for i, c in enumerate(chunks, 1):
        print(f"\n[{i}] score={c['score']:.4f}  fonte={c['source']}  pág={c['page']}")
        if c["section"]:
            print(f"     seção: {c['section']}")
        if c["course_code"]:
            print(f"     disciplina: {c['course_code']}")
        print(f"     {c['excerpt']}")

    print("\n--- Formatado para prompt ---")
    print(format_chunks_for_prompt(chunks, max_chars=1500))