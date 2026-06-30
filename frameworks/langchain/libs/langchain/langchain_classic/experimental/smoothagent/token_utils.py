"""Lightweight token-count helpers used by the lookahead strategies.

The strategies in this module operate on length thresholds (soft / hard
limits, recent-K windows). They need a deterministic, dependency-free
estimator so that unit tests do not require a tokenizer.

The default estimator approximates token counts as ``ceil(len(text) / 4)``,
which is the standard rule-of-thumb for BPE tokenizers.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

Tokenizer = Callable[[str], int]


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _message_text(message: Any) -> str:
    """Extract the textual payload from a message-like object.

    Supports langchain ``BaseMessage`` instances, plain dicts produced by
    OpenAI-style callers, and bare strings. Tool / observation contents
    that are not strings are coerced via ``repr`` so that token counts
    remain stable across representations.
    """
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(repr(item))
            else:
                parts.append(repr(item))
        return "\n".join(parts)
    if isinstance(content, str):
        return content
    return repr(content) if content is not None else ""


def count_message_tokens(
    message: Any,
    tokenizer: Tokenizer | None = None,
) -> int:
    """Approximate the token count of a single message."""
    text = _message_text(message)
    if tokenizer is not None:
        return int(tokenizer(text))
    return _approx_token_count(text)


def count_tokens(
    messages: list[Any],
    tokenizer: Tokenizer | None = None,
) -> int:
    """Approximate the token count of a message list."""
    return sum(count_message_tokens(m, tokenizer) for m in messages)
