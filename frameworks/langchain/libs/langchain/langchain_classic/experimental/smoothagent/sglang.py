"""SGLang OpenAI-compatible helpers for SmoothAgent lookahead calls."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import requests

from langchain_classic.experimental.smoothagent.request_meta import (
    LookaheadRequestMeta,
    build_extra_body,
)

ExtraBodyFactory = Callable[[list[Any]], Mapping[str, Any]]


@dataclass
class SGLangSummaryClient:
    """Callable summary client backed by SGLang's chat-completions API."""

    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 60.0
    max_tokens: int = 128
    temperature: float = 0.0
    lookahead_group_id: str | None = None
    extra_body: Mapping[str, Any] | ExtraBodyFactory | None = None

    def __call__(self, messages: list[Any]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_message(message) for message in messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        payload.update(dict(self._extra_body(messages)))
        response = requests.post(
            self._chat_completions_url(),
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"].get("content", ""))

    def _chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _extra_body(self, messages: list[Any]) -> Mapping[str, Any]:
        if callable(self.extra_body):
            return self.extra_body(messages)
        if self.extra_body is not None:
            return self.extra_body
        return build_extra_body(
            LookaheadRequestMeta(
                is_lookahead=True,
                request_class="bg",
                priority_class="be",
                lookahead_group_id=self.lookahead_group_id,
            )
        )


def _to_openai_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        role = str(message.get("role", message.get("type", "user")))
        return {"role": role, "content": message.get("content", "")}
    role = getattr(message, "role", None) or getattr(message, "type", None) or "user"
    content = getattr(message, "content", message)
    return {"role": str(role), "content": content}
