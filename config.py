"""Application settings loaded from environment and optional ``.env`` file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for RAG: OpenRouter for chat, local embeddings for vectors."""

    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_model: str
    openrouter_http_referer: str
    openrouter_app_name: str
    embedding_model_name: str
    chroma_persist_dir: str
    pdf_base_url: str
    use_contextual_compression: bool
    compression_fetch_k: int | None


def load_settings() -> Settings:
    """Load ``.env`` and build :class:`Settings`. Raises if ``OPENROUTER_API_KEY`` is missing."""
    load_dotenv(find_dotenv(), override=False)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        msg = "OPENROUTER_API_KEY is required (set in environment or .env)"
        raise ValueError(msg)
    _cfk = os.environ.get("COMPRESSION_FETCH_K", "").strip()
    return Settings(
        openrouter_api_key=key,
        openrouter_base_url=os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/"),
        openrouter_model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        openrouter_http_referer=os.environ.get(
            "OPENROUTER_HTTP_REFERER", "http://localhost"
        ),
        openrouter_app_name=os.environ.get("OPENROUTER_APP_NAME", "langchain-rag"),
        embedding_model_name=os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chroma_persist_dir=os.environ.get("CHROMA_PERSIST_DIR", "docs/chroma"),
        pdf_base_url=os.environ.get(
            "PDF_BASE_URL",
            "https://raw.githubusercontent.com/masrik-dev/LangChain-Chat-with-Your-Data/main/docs/cs229_lectures",
        ),
        use_contextual_compression=os.environ.get(
            "USE_CONTEXTUAL_COMPRESSION", ""
        ).lower()
        in ("1", "true", "yes"),
        compression_fetch_k=int(_cfk) if _cfk else None,
    )
