"""Retrieval-augmented QA chain."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter

if TYPE_CHECKING:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_openai import ChatOpenAI


def resolve_compression_fetch_k(
    retrieval_k: int,
    compression_fetch_k: int | None,
) -> int:
    """Align with ``EmbeddingsFilter`` pipeline: how many chunks the base retriever pulls before filtering."""
    fetch_k = compression_fetch_k if compression_fetch_k is not None else max(retrieval_k * 2, 6)
    return max(fetch_k, retrieval_k)


def build_retrieval_qa(
    vectordb: Chroma,
    llm: ChatOpenAI,
    embedding: HuggingFaceEmbeddings,
    *,
    retrieval_k: int = 3,
    use_contextual_compression: bool = False,
    compression_fetch_k: int | None = None,
) -> RetrievalQA:
    """Stuff-documents RAG. Optional contextual compression re-ranks retrieved chunks with the same embeddings model."""
    if use_contextual_compression:
        fetch_k = resolve_compression_fetch_k(retrieval_k, compression_fetch_k)
        base_retriever = vectordb.as_retriever(search_kwargs={"k": fetch_k})
        compressor = EmbeddingsFilter(embeddings=embedding, k=retrieval_k)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
    else:
        retriever = vectordb.as_retriever(search_kwargs={"k": retrieval_k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )
