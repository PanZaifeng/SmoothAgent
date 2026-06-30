"""SGLang OpenAI-compatible helpers for the MiniAgent integration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import httpx

RequestClass = Literal["fg", "bg"]


@dataclass(slots=True)
class SmoothAgentRequestMeta:
    request_class: RequestClass
    lookahead_group_id: str
    is_lookahead: bool | None = None
    priority_class: str | None = None
    arrival_time_ms: float | None = None
    commit_deadline_ms: float | None = None
    slo_ttft_ms: float | None = None
    slo_tbt_ms: float | None = None


@dataclass(slots=True)
class ChatResult:
    ttft_ms: float
    duration_ms: float
    text: str
    meta: dict[str, Any]
    server_rid: str


def build_extra_body(meta: SmoothAgentRequestMeta) -> dict[str, Any]:
    background = meta.request_class == "bg"
    body: dict[str, Any] = {
        "request_class": meta.request_class,
        "is_lookahead": background if meta.is_lookahead is None else meta.is_lookahead,
        "priority_class": meta.priority_class or ("be" if background else "lc"),
        "lookahead_group_id": meta.lookahead_group_id,
        "arrival_time_ms": meta.arrival_time_ms or time.time() * 1000.0,
    }
    for key in ("commit_deadline_ms", "slo_ttft_ms", "slo_tbt_ms"):
        value = getattr(meta, key)
        if value is not None:
            body[key] = value
    return body


def _normalise_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _extract_text_delta(obj: dict[str, Any]) -> str:
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        return content if isinstance(content, str) else ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else ""
    text = first.get("text")
    return text if isinstance(text, str) else ""


def stream_chat_completion(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    request_meta: SmoothAgentRequestMeta,
    max_tokens: int,
    timeout: float,
) -> ChatResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    payload.update(build_extra_body(request_meta))
    t0 = perf_counter()
    ttft_ms = -1.0
    text_parts: list[str] = []
    meta: dict[str, Any] = {}
    server_rid = ""
    with client.stream(
        "POST",
        f"{_normalise_base_url(base_url)}/chat/completions",
        json=payload,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                meta = obj.get("meta_info") or meta
                server_rid = str(meta.get("id") or server_rid)
                delta = _extract_text_delta(obj)
                if delta:
                    if ttft_ms < 0:
                        ttft_ms = (perf_counter() - t0) * 1000.0
                    text_parts.append(delta)
    duration_ms = (perf_counter() - t0) * 1000.0
    return ChatResult(ttft_ms, duration_ms, "".join(text_parts), meta, server_rid)


def chat_completion(
    client: httpx.Client,
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    request_meta: SmoothAgentRequestMeta,
    max_tokens: int,
    timeout: float,
) -> ChatResult:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    payload.update(build_extra_body(request_meta))
    t0 = perf_counter()
    resp = client.post(
        f"{_normalise_base_url(base_url)}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    duration_ms = (perf_counter() - t0) * 1000.0
    obj = resp.json()
    text = _extract_text_delta(obj) if isinstance(obj, dict) else ""
    meta = obj.get("meta_info") or {} if isinstance(obj, dict) else {}
    return ChatResult(duration_ms, duration_ms, text, meta, str(meta.get("id") or ""))
