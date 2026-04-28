"""Chat model client routed through OpenRouter (OpenAI-compatible HTTP API)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from config import Settings


def build_chat_llm(settings: Settings) -> ChatOpenAI:
    """Return a chat model that calls OpenRouter using ``OPENROUTER_API_KEY``."""
    return ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        temperature=0,
        default_headers={
            "HTTP-Referer": settings.openrouter_http_referer,
            "X-Title": settings.openrouter_app_name,
        },
    )
