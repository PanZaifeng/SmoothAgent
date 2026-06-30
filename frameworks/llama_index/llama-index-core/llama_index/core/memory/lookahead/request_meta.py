"""SGLang/OpenAI-compatible request metadata for lookahead calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

PriorityClass = Literal["lc", "be"]
RequestClass = Literal["fg", "bg"]


@dataclass
class LookaheadRequestMeta:
    """Metadata carried in OpenAI ``extra_body`` for SmoothAgent scheduling."""

    is_lookahead: bool = False
    request_class: Optional[RequestClass] = None
    priority_class: PriorityClass = "lc"
    lookahead_group_id: Optional[str] = None
    commit_deadline_ms: Optional[float] = None
    slo_ttft_ms: Optional[float] = None
    slo_tbt_ms: Optional[float] = None
    arrival_time_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.priority_class not in ("lc", "be"):
            raise ValueError(
                f"priority_class must be 'lc' or 'be', got {self.priority_class!r}"
            )
        if self.request_class not in (None, "fg", "bg"):
            raise ValueError(
                f"request_class must be 'fg' or 'bg', got {self.request_class!r}"
            )
        if self.is_lookahead and self.priority_class != "be":
            raise ValueError("is_lookahead=True requires priority_class='be'")


def build_extra_body(meta: LookaheadRequestMeta) -> dict[str, Any]:
    """Build top-level OpenAI ``extra_body`` fields consumed by SGLang."""
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


def is_be_request(extra_body: Optional[dict[str, Any]]) -> bool:
    """Return whether ``extra_body`` marks a best-effort lookahead request."""
    if not extra_body:
        return False
    return extra_body.get("is_lookahead") is True or (
        extra_body.get("priority_class") == "be"
    )
