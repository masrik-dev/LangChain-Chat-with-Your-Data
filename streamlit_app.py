"""Streamlit UI PDF RAG (OpenRouter + local embeddings + Chroma)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"(?i).*langchain.*deprecated.*",
)
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st

from chains import build_retrieval_qa
from compression import build_compression_preview, render_compression_viz
from config import Settings, load_settings
from llm import build_chat_llm
from vectorstore import build_embeddings, ensure_vectorstore


@st.cache_resource(show_spinner="Loading embeddings and index…")
def _load_index(settings: Settings, rebuild_token: int):
    """Cache embeddings and Chroma; bump ``rebuild_token`` in session state to force re-ingest."""
    embedding = build_embeddings(settings)
    persist = Path(settings.chroma_persist_dir)
    force = rebuild_token > 0
    vectordb = ensure_vectorstore(settings, embedding, persist, force_rebuild=force)
    return vectordb, embedding


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="CS229 RAG", layout="wide")
    st.title("Chat with lecture notes")

    try:
        settings = load_settings()
    except ValueError as err:
        st.error(str(err))
        st.stop()

    if "rebuild_token" not in st.session_state:
        st.session_state.rebuild_token = 0

    with st.sidebar:
        st.subheader("Settings")
        st.text(f"Model: {settings.openrouter_model}")
        st.text(f"Index: {settings.chroma_persist_dir}")
        retrieval_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=3)
        use_compression = st.checkbox(
            "Contextual compression",
            value=settings.use_contextual_compression,
            help="Re-rank retrieved chunks with EmbeddingsFilter (same local model as the index; no extra LLM cost).",
        )
        if st.button("Rebuild index"):
            # Release Chroma file handles on Windows before deleting persist_dir.
            st.cache_resource.clear()
            st.session_state.rebuild_token = st.session_state.get("rebuild_token", 0) + 1
            st.rerun()

    vectordb, embedding = _load_index(settings, st.session_state.rebuild_token)

    with st.form("ask"):
        query = st.text_input("Question", placeholder="What is machine learning?")
        submitted = st.form_submit_button("Ask")

    if submitted and query.strip():
        preview_rows = None
        preview_fetch = 0
        with st.spinner("Generating answer…"):
            llm = build_chat_llm(settings)
            qa = build_retrieval_qa(
                vectordb,
                llm,
                embedding,
                retrieval_k=retrieval_k,
                use_contextual_compression=use_compression,
                compression_fetch_k=settings.compression_fetch_k,
            )
            result = qa.invoke({"query": query.strip()})
            if use_compression:
                preview_rows, preview_fetch = build_compression_preview(
                    vectordb,
                    embedding,
                    query.strip(),
                    retrieval_k,
                    settings.compression_fetch_k,
                )
        st.markdown(result.get("result", ""))
        if use_compression and preview_rows is not None:
            render_compression_viz(preview_rows, preview_fetch, retrieval_k)
        st.subheader("Sources")
        for i, doc in enumerate(result.get("source_documents", [])):
            with st.expander(f"[{i + 1}] {doc.metadata.get('source', 'Unknown')}"):
                st.text(doc.page_content[:4000])


if __name__ == "__main__":
    main()
