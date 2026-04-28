"""Load and chunk PDF documents for retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from config import Settings


def load_pdf_documents(settings: Settings) -> list[Document]:
    """Load CS229 lecture PDFs (including duplicate Lecture 01) from configured source."""
    return load_pdf_documents_from_base(settings.pdf_base_url)


def load_pdf_documents_from_base(pdf_base_url: str) -> list[Document]:
    """Load CS229 lecture PDFs from either a local directory path or URL base path."""
    base = pdf_base_url.strip()
    is_remote = urlparse(base).scheme in {"http", "https"}
    filenames = [
        "MachineLearning-Lecture01.pdf",
        "MachineLearning-Lecture01.pdf",
        "MachineLearning-Lecture02.pdf",
        "MachineLearning-Lecture03.pdf",
    ]
    sources = (
        [f"{base.rstrip('/')}/{name}" for name in filenames]
        if is_remote
        else [str(Path(base) / name) for name in filenames]
    )
    loaders = [
        PyPDFLoader(source) for source in sources
    ]
    docs: list[Document] = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents with recursive character chunking (1500 / 150 overlap)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    return splitter.split_documents(documents)
