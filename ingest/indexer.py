# ingest/indexer.py
# Indexador do corpus da UFCG
#
# Pipeline:
#   1. Varre data/raw/ procurando PDFs e HTMLs
#   2. Extrai texto com metadados (source, page, section, course_code)
#   3. Aplica chunking semântico com sobreposição
#   4. Gera embeddings com bge-m3 (HuggingFace)
#   5. Salva índice FAISS em data/faiss_index/
#
# Uso:
#   python ingest/indexer.py                  # indexa tudo em data/raw/
#   python ingest/indexer.py --file foo.pdf   # indexa um arquivo específico
#   python ingest/indexer.py --reset          # apaga índice anterior e reindexa

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Iterator

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

RAW_DIR        = Path(os.getenv("RAW_DIR",        "data/raw"))
FAISS_DIR      = Path(os.getenv("FAISS_INDEX_PATH","data/faiss_index"))
EMBEDDING_MODEL= os.getenv("EMBEDDING_MODEL",     "BAAI/bge-m3")

CHUNK_SIZE     = 512    # caracteres por chunk
CHUNK_OVERLAP  = 80     # sobreposição entre chunks consecutivos
BATCH_SIZE     = 64     # documentos por batch de embedding

# Padrão para detectar código de disciplina no texto
_COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,6}\d{3,5})\b")

# ---------------------------------------------------------------------------
# Extratores de texto
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> Iterator[Document]:
    """
    Extrai texto de PDF página a página usando pypdf.
    Cada página vira um Document com metadados de source e page.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Instale pypdf: pip install pypdf")

    reader = PdfReader(str(path))
    logger.info("[indexer] PDF %s — %d páginas", path.name, len(reader.pages))

    current_section = ""
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _clean_text(text)
        if not text.strip():
            continue

        # Detecta título de seção (linha em MAIÚSCULAS com 10–80 chars)
        section_match = re.search(r"^([A-ZÁÉÍÓÚÃÕ][A-ZÁÉÍÓÚÃÕ\s]{9,79})$", text, re.MULTILINE)
        if section_match:
            current_section = section_match.group(1).strip()

        # Detecta código de disciplina mais frequente na página
        codes = _COURSE_CODE_RE.findall(text)
        course_code = max(set(codes), key=codes.count) if codes else ""

        yield Document(
            page_content=text,
            metadata={
                "source":      path.name,
                "source_path": str(path),
                "page":        page_num,
                "section":     current_section,
                "course_code": course_code,
                "file_type":   "pdf",
            },
        )


def _extract_html(path: Path) -> Iterator[Document]:
    """
    Extrai texto de arquivo HTML local usando BeautifulSoup.
    Remove navegação, scripts e estilos.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Instale beautifulsoup4: pip install beautifulsoup4")

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Remove elementos não-textuais
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Tenta extrair por seções (h2/h3)
    sections = soup.find_all(["h2", "h3"])
    if sections:
        for heading in sections:
            section_title = heading.get_text(strip=True)
            content_parts = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ("h2", "h3"):
                    break
                content_parts.append(sibling.get_text(separator=" ", strip=True))

            text = _clean_text(" ".join(content_parts))
            if not text.strip():
                continue

            codes = _COURSE_CODE_RE.findall(text)
            course_code = max(set(codes), key=codes.count) if codes else ""

            yield Document(
                page_content=text,
                metadata={
                    "source":      path.name,
                    "source_path": str(path),
                    "page":        None,
                    "section":     section_title,
                    "course_code": course_code,
                    "file_type":   "html",
                },
            )
    else:
        # Sem estrutura de seções: extrai tudo como um bloco
        text = _clean_text(soup.get_text(separator="\n", strip=True))
        if text.strip():
            codes = _COURSE_CODE_RE.findall(text)
            course_code = max(set(codes), key=codes.count) if codes else ""
            yield Document(
                page_content=text,
                metadata={
                    "source":      path.name,
                    "source_path": str(path),
                    "page":        None,
                    "section":     "",
                    "course_code": course_code,
                    "file_type":   "html",
                },
            )


def _extract_txt(path: Path) -> Iterator[Document]:
    """Extrai texto puro de arquivo .txt."""
    text = _clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    if text.strip():
        codes = _COURSE_CODE_RE.findall(text)
        course_code = max(set(codes), key=codes.count) if codes else ""
        yield Document(
            page_content=text,
            metadata={
                "source":      path.name,
                "source_path": str(path),
                "page":        None,
                "section":     "",
                "course_code": course_code,
                "file_type":   "txt",
            },
        )


def _clean_text(text: str) -> str:
    """Remove artefatos comuns de extração de PDF/HTML."""
    # Remove caracteres de controle (exceto newline)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    # Colapsa espaços múltiplos
    text = re.sub(r" {2,}", " ", text)
    # Remove linhas com apenas números (cabeçalhos/rodapés de página)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Colapsa linhas em branco múltiplas
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Carrega documentos brutos
# ---------------------------------------------------------------------------

def load_documents(raw_dir: Path, single_file: Path | None = None) -> list[Document]:
    """
    Varre raw_dir e extrai todos os documentos suportados.
    Se single_file for fornecido, processa apenas esse arquivo.
    """
    extractors = {
        ".pdf":  _extract_pdf,
        ".html": _extract_html,
        ".htm":  _extract_html,
        ".txt":  _extract_txt,
    }

    files: list[Path] = []
    if single_file:
        files = [single_file]
    else:
        for ext in extractors:
            files.extend(raw_dir.rglob(f"*{ext}"))
        files = sorted(set(files))

    if not files:
        logger.warning("[indexer] nenhum arquivo encontrado em %s", raw_dir)
        return []

    logger.info("[indexer] %d arquivo(s) encontrado(s)", len(files))
    docs: list[Document] = []

    for path in files:
        ext = path.suffix.lower()
        extractor = extractors.get(ext)
        if not extractor:
            logger.warning("[indexer] formato não suportado: %s", path.name)
            continue
        try:
            extracted = list(extractor(path))
            logger.info("[indexer]   %s → %d páginas/seções", path.name, len(extracted))
            docs.extend(extracted)
        except Exception as exc:
            logger.error("[indexer] erro ao extrair %s: %s", path.name, exc)

    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Divide os documentos em chunks com sobreposição.
    Usa RecursiveCharacterTextSplitter com separadores em português.
    Preserva todos os metadados do documento original e adiciona chunk_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[Document] = []
    for doc in docs:
        split_docs = splitter.split_documents([doc])
        for i, chunk in enumerate(split_docs):
            # Adiciona chunk_id único e índice do chunk no documento original
            chunk.metadata["chunk_id"] = str(uuid.uuid4())[:8]
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(split_docs)
            chunks.append(chunk)

    logger.info("[indexer] %d docs → %d chunks (size=%d, overlap=%d)",
                len(docs), len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


# ---------------------------------------------------------------------------
# Embeddings e índice FAISS
# ---------------------------------------------------------------------------

def load_embeddings() -> HuggingFaceEmbeddings:
    """Carrega o modelo bge-m3. Detecta GPU automaticamente."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    logger.info("[indexer] carregando embeddings %s no device=%s", EMBEDDING_MODEL, device)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": BATCH_SIZE},
    )


def build_faiss_index(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    output_dir: Path,
    reset: bool = False,
) -> FAISS:
    """
    Cria ou atualiza o índice FAISS.

    Se reset=False e já existir um índice em output_dir,
    carrega o índice existente e adiciona os novos chunks.
    Se reset=True, apaga o índice anterior.
    """
    if reset and output_dir.exists():
        logger.info("[indexer] apagando índice anterior em %s", output_dir)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    index_exists = (output_dir / "index.faiss").exists()

    if index_exists and not reset:
        logger.info("[indexer] índice existente encontrado — adicionando %d chunks", len(chunks))
        vectorstore = FAISS.load_local(
            str(output_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vectorstore.add_documents(chunks)
    else:
        logger.info("[indexer] criando novo índice com %d chunks", len(chunks))
        t0 = time.time()

        # Processa em batches para mostrar progresso
        vectorstore = None
        for i in range(0, len(chunks), BATCH_SIZE * 4):
            batch = chunks[i: i + BATCH_SIZE * 4]
            batch_num = i // (BATCH_SIZE * 4) + 1
            total_batches = (len(chunks) + BATCH_SIZE * 4 - 1) // (BATCH_SIZE * 4)
            logger.info("[indexer] batch %d/%d (%d chunks)...", batch_num, total_batches, len(batch))

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)

        elapsed = time.time() - t0
        logger.info("[indexer] índice criado em %.1fs", elapsed)

    vectorstore.save_local(str(output_dir))
    logger.info("[indexer] índice salvo em %s (%d vetores)", output_dir, vectorstore.index.ntotal)

    # Salva metadados do índice para auditoria
    _save_index_metadata(output_dir, chunks)

    return vectorstore


def _save_index_metadata(output_dir: Path, chunks: list[Document]) -> None:
    """Salva um arquivo JSON com estatísticas do índice para documentação."""
    sources: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "?")
        sources[src] = sources.get(src, 0) + 1

    metadata = {
        "total_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": sources,
    }

    meta_path = output_dir / "index_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    logger.info("[indexer] metadados salvos em %s", meta_path)


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_dir: Path = RAW_DIR,
    faiss_dir: Path = FAISS_DIR,
    single_file: Path | None = None,
    reset: bool = False,
) -> dict:
    """
    Executa o pipeline completo de ingestão.

    Returns:
        Dicionário com estatísticas: n_files, n_docs, n_chunks, elapsed
    """
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("Iniciando pipeline de ingestão")
    logger.info("  raw_dir    : %s", raw_dir)
    logger.info("  faiss_dir  : %s", faiss_dir)
    logger.info("  single_file: %s", single_file)
    logger.info("  reset      : %s", reset)
    logger.info("=" * 60)

    # 1. Carrega documentos
    docs = load_documents(raw_dir, single_file)
    if not docs:
        logger.error("[indexer] nenhum documento carregado. Abortando.")
        return {"n_files": 0, "n_docs": 0, "n_chunks": 0, "elapsed": 0}

    n_files = len(set(d.metadata["source"] for d in docs))

    # 2. Chunking
    chunks = chunk_documents(docs)

    # 3. Embeddings
    embeddings = load_embeddings()

    # 4. Índice FAISS
    build_faiss_index(chunks, embeddings, faiss_dir, reset=reset)

    elapsed = time.time() - t_start
    stats = {
        "n_files":  n_files,
        "n_docs":   len(docs),
        "n_chunks": len(chunks),
        "elapsed":  round(elapsed, 1),
    }

    logger.info("=" * 60)
    logger.info("Pipeline concluído em %.1fs", elapsed)
    logger.info("  Arquivos  : %d", stats["n_files"])
    logger.info("  Páginas   : %d", stats["n_docs"])
    logger.info("  Chunks    : %d", stats["n_chunks"])
    logger.info("=" * 60)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Indexa documentos da UFCG no FAISS com embeddings bge-m3"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR,
        help=f"Diretório com os documentos brutos (padrão: {RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=FAISS_DIR,
        help=f"Diretório de saída do índice FAISS (padrão: {FAISS_DIR})",
    )
    parser.add_argument(
        "--file", type=Path, default=None,
        help="Indexa apenas um arquivo específico",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Apaga o índice existente e reindexa tudo do zero",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Tamanho de cada chunk em caracteres (padrão: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=CHUNK_OVERLAP,
        help=f"Sobreposição entre chunks (padrão: {CHUNK_OVERLAP})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Sobrescreve globals com args da CLI
    CHUNK_SIZE    = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap

    stats = run_pipeline(
        raw_dir=args.raw_dir,
        faiss_dir=args.output_dir,
        single_file=args.file,
        reset=args.reset,
    )

    print(f"\nResumo: {stats['n_chunks']} chunks de {stats['n_files']} arquivo(s) em {stats['elapsed']}s")