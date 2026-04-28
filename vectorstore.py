"""Embedding model and Chroma vector store."""

from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from config import Settings


def _rmtree_robust(path: Path) -> None:
    """Remove a directory tree on Windows where Chroma may still hold file handles briefly."""
    if not path.exists():
        return
    gc.collect()
    time.sleep(0.2)
    for attempt in range(8):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.4 * (attempt + 1))
    shutil.rmtree(path, ignore_errors=True)


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Local sentence-transformers embeddings (OpenRouter does not expose an embeddings API)."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)


def _persist_has_chroma(persist_dir: Path) -> bool:
    """Return True if ``persist_dir`` looks like a Chroma 0.4+ SQLite store."""
    return (persist_dir / "chroma.sqlite3").is_file()


def open_vectorstore(persist_dir: Path, embedding: HuggingFaceEmbeddings) -> Chroma:
    """Open an existing on-disk Chroma index."""
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding,
    )


def build_vectorstore(
    splits: list[Document],
    embedding: HuggingFaceEmbeddings,
    persist_dir: Path,
) -> Chroma:
    """Persist Chroma under ``persist_dir``, replacing any existing store."""
    if persist_dir.exists():
        _rmtree_robust(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=str(persist_dir),
    )


def ensure_vectorstore(
    settings: Settings,
    embedding: HuggingFaceEmbeddings,
    persist_dir: Path,
    *,
    force_rebuild: bool = False,
) -> Chroma:
    """Load Chroma from disk when present; otherwise ingest PDFs. Optionally wipe and rebuild."""
    if force_rebuild and persist_dir.exists():
        gc.collect()
        _rmtree_robust(persist_dir)
    if persist_dir.exists() and _persist_has_chroma(persist_dir):
        try:
            return open_vectorstore(persist_dir, embedding)
        except Exception:
            _rmtree_robust(persist_dir)
    from documents import load_pdf_documents, split_documents

    docs = load_pdf_documents(settings)
    splits = split_documents(docs)
    return build_vectorstore(splits, embedding, persist_dir)