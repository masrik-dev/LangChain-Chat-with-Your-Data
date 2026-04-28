"""Load and chunk PDF documents for retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from config import Settings


def load_pdf_documents(settings: Settings) -> list[Document]:
    """Load CS229 lecture PDFs (including duplicate Lecture 01) from configured base URL."""
    return load_pdf_documents_from_base(settings.pdf_base_url)


def load_pdf_documents_from_base(pdf_base_url: str) -> list[Document]:
    """Load CS229 lecture PDFs from a raw URL base path."""
    base = pdf_base_url.rstrip("/")
    loaders = [
        PyPDFLoader(f"{base}/MachineLearning-Lecture01.pdf"),
        PyPDFLoader(f"{base}/MachineLearning-Lecture01.pdf"),
        PyPDFLoader(f"{base}/MachineLearning-Lecture02.pdf"),
        PyPDFLoader(f"{base}/MachineLearning-Lecture03.pdf"),
    ]
    docs: list[Document] = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents with recursive character chunking (1500 / 150 overlap)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    return splitter.split_documents(documents)
