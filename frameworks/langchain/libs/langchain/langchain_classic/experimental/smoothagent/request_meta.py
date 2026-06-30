"""Outbound-request metadata helpers used to mark BE / lookahead calls.

These are used by :class:`LookaheadRuntime` and by integration code that
forwards lookahead-stream LLM calls to an SGLang serving instance. The
shape of :class:`LookaheadRequestMeta` mirrors the schema added on the
SGLang side (``python/sglang/srt/managers/smoothagent/smoothagent_meta.py``)
so that ``build_extra_body`` produces a payload the backend can consume.

Plain Python dicts are used as the wire format because they survive any
JSON-based ``extra_body`` channel — both the OpenAI-compatible HTTP API
and SGLang's ``GenerateReqInput`` accept them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PriorityClass = Literal["lc", "be"]
RequestClass = Literal["fg", "bg"]


@dataclass
class LookaheadRequestMeta:
    """Per-request metadata carried alongside the prompt.

    Attributes:
        is_lookahead: ``True`` for best-effort lookahead requests.
        request_class: SGLang request class (``"fg"`` or ``"bg"``).
        priority_class: Coarse priority bucket (``"lc"`` or ``"be"``).
        lookahead_group_id: Logical id grouping a lookahead transform
            with its eventual commit request, used for prefix-cache
            reuse on the serving side.
        commit_deadline_ms: Wall-clock deadline (in ms) by which the
            corresponding commit is expected.
        slo_ttft_ms: TTFT SLO for the request (LC only).
        slo_tbt_ms: TBT SLO for the request (LC only).
        arrival_time_ms: Optional client-side arrival timestamp; the
            scheduler uses this to compute slack.
    """

    is_lookahead: bool = False
    request_class: RequestClass | None = None
    priority_class: PriorityClass = "lc"
    lookahead_group_id: str | None = None
    commit_deadline_ms: float | None = None
    slo_ttft_ms: float | None = None
    slo_tbt_ms: float | None = None
    arrival_time_ms: float | None = None

    def __post_init__(self) -> None:
        if self.priority_class not in ("lc", "be"):
            msg = f"priority_class must be 'lc' or 'be', got {self.priority_class!r}"
            raise ValueError(msg)
        if self.request_class not in (None, "fg", "bg"):
            msg = f"request_class must be 'fg' or 'bg', got {self.request_class!r}"
            raise ValueError(msg)
        if self.is_lookahead and self.priority_class != "be":
            msg = "is_lookahead=True requires priority_class='be'"
            raise ValueError(msg)


def build_extra_body(meta: LookaheadRequestMeta) -> dict[str, Any]:
    """Serialize ``meta`` into the dict shape expected by SGLang's API."""
    payload: dict[str, Any] = {
        "is_lookahead": bool(meta.is_lookahead),
        "priority_class": meta.priority_class,
    }
    if meta.request_class is not None:
        payload["request_class"] = meta.request_class
    if meta.lookahead_group_id is not None:
        payload["lookahead_group_id"] = str(meta.lookahead_group_id)
    if meta.commit_deadline_ms is not None:
        payload["commit_deadline_ms"] = float(meta.commit_deadline_ms)
    if meta.slo_ttft_ms is not None:
        payload["slo_ttft_ms"] = float(meta.slo_ttft_ms)
    if meta.slo_tbt_ms is not None:
        payload["slo_tbt_ms"] = float(meta.slo_tbt_ms)
    if meta.arrival_time_ms is not None:
        payload["arrival_time_ms"] = float(meta.arrival_time_ms)
    return payload


def is_be_request(extra_body: dict[str, Any] | None) -> bool:
    """Return ``True`` if ``extra_body`` describes a best-effort request."""
    if not extra_body:
        return False
    if extra_body.get("is_lookahead") is True:
        return True
    return extra_body.get("priority_class") == "be"
