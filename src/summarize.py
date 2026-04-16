"""LLM summarization helpers for digest article cards."""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

DEFAULT_MODEL = "gpt-4.1-mini"
MAX_INPUT_CHARS = 6000


def _build_prompt(title: str, source: Optional[str], content: str) -> str:
    source_text = source or "Unknown source"
    trimmed = content[:MAX_INPUT_CHARS]
    return (
        "Summarize the following news article for a user-personalized digest. "
        "Write 2 concise sentences, neutral tone, include the key development and why it matters. "
        "Avoid speculation and avoid repeating the title.\n\n"
        f"Title: {title}\n"
        f"Source: {source_text}\n"
        f"Article:\n{trimmed}"
    )


def summarize_article(
    *,
    title: str,
    source: Optional[str],
    content: str,
    model_name: str = DEFAULT_MODEL,
) -> tuple[str, str]:
    """Return summary text and model used."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        fallback = "OpenAI API key missing. Set OPENAI_API_KEY to enable LLM summaries."
        return fallback, "fallback:no_api_key"

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model_name,
        input=_build_prompt(title=title, source=source, content=content),
        temperature=0.2,
        max_output_tokens=120,
    )
    text = (response.output_text or "").strip()
    if not text:
        text = "Summary unavailable for this article right now."
    return text, model_name
