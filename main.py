"""CLI entry: ingest PDFs, build Chroma, run sample retrieval and QA via OpenRouter."""

from __future__ import annotations

from pathlib import Path

from chains import build_retrieval_qa
from config import load_settings
from llm import build_chat_llm
from vectorstore import build_embeddings, ensure_vectorstore


def run() -> None:
    """Execute the demo pipeline."""
    settings = load_settings()
    embedding = build_embeddings(settings)
    persist_path = Path(settings.chroma_persist_dir)
    vectordb = ensure_vectorstore(settings, embedding, persist_path, force_rebuild=False)

    question = "is there an email i can ask for help"
    for doc in vectordb.similarity_search(question, k=3):
        print(doc.page_content[:200], "...\n")

    llm = build_chat_llm(settings)
    qa = build_retrieval_qa(
        vectordb,
        llm,
        embedding,
        use_contextual_compression=settings.use_contextual_compression,
        compression_fetch_k=settings.compression_fetch_k,
    )
    result = qa.invoke({"query": "What is machine learning?"})
    print("ANSWER:\n", result["result"])
    for i, doc in enumerate(result["source_documents"]):
        print(f"[{i + 1}] {doc.metadata.get('source', 'Unknown')}")


if __name__ == "__main__":
    run()
