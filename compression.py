"""Streamlit visualization for EmbeddingsFilter-style contextual compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st
from langchain_community.utils.math import cosine_similarity
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter

from chains import resolve_compression_fetch_k

if TYPE_CHECKING:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma


@dataclass
class CompressionRow:
    """One retrieved chunk and whether it survives compression."""

    rank: int
    source: str
    preview: str
    similarity: float
    kept: bool


def build_compression_preview(
    vectordb: Chroma,
    embedding: HuggingFaceEmbeddings,
    query: str,
    retrieval_k: int,
    compression_fetch_k: int | None,
    *,
    preview_chars: int = 220,
) -> tuple[list[CompressionRow], int]:
    """Run the same fetch-then-filter logic as the QA chain and score chunks vs the query."""
    fetch_k = resolve_compression_fetch_k(retrieval_k, compression_fetch_k)
    base = vectordb.as_retriever(search_kwargs={"k": fetch_k})
    raw_docs = base.invoke(query)
    if not raw_docs:
        return [], fetch_k

    compressor = EmbeddingsFilter(embeddings=embedding, k=retrieval_k)
    compressed = list(compressor.compress_documents(raw_docs, query))
    kept_texts = {d.page_content for d in compressed}

    texts = [d.page_content for d in raw_docs]
    qv = embedding.embed_query(query)
    dvs = embedding.embed_documents(texts)
    sim = cosine_similarity([qv], dvs)[0]

    rows: list[CompressionRow] = []
    order = sorted(range(len(raw_docs)), key=lambda i: float(sim[i]), reverse=True)
    for rank, i in enumerate(order, start=1):
        d = raw_docs[i]
        pc = d.page_content
        prev = (pc[:preview_chars] + "…") if len(pc) > preview_chars else pc
        rows.append(
            CompressionRow(
                rank=rank,
                source=str(d.metadata.get("source", ""))[:100],
                preview=prev,
                similarity=float(sim[i]),
                kept=pc in kept_texts,
            )
        )
    return rows, fetch_k


def render_compression_viz(
    rows: list[CompressionRow],
    fetch_k: int,
    retrieval_k: int,
) -> None:
    """Draw metrics, similarity chart, and ranked table."""
    st.subheader("Compression")
    if not rows:
        st.info("No chunks retrieved for this query.")
        return

    n_kept = sum(1 for r in rows if r.kept)
    c1, c2, c3 = st.columns(3)
    c1.metric("Fetched (k)", len(rows))
    c2.metric("Kept for LLM", n_kept)
    c3.metric("Dropped", len(rows) - n_kept)
    st.caption(
        f"Base retriever uses k={fetch_k}; EmbeddingsFilter keeps top {retrieval_k} by cosine similarity to the question."
    )

    chart_df = pd.DataFrame(
        {
            "chunk": [f"#{r.rank}" for r in rows],
            "similarity": [r.similarity for r in rows],
            "kept": [r.kept for r in rows],
        }
    )
    st.bar_chart(chart_df.set_index("chunk")["similarity"], height=220)

    table = pd.DataFrame(
        {
            "Rank": [r.rank for r in rows],
            "Kept": ["yes" if r.kept else "no" for r in rows],
            "Similarity": [round(r.similarity, 4) for r in rows],
            "Source": [r.source for r in rows],
            "Preview": [r.preview for r in rows],
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)
