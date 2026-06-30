"""Per-request lookahead metadata for the SmoothAgent SGLang integration.

This module is the SGLang-side counterpart of the LangChain helper at
``langchain_classic.experimental.smoothagent.request_meta``. The two
modules use identical wire-format field names so that an ``extra_body``
payload built on the agent side can be ingested verbatim here.

The dataclass fields are intentionally optional and default to a
"latency-critical, non-lookahead" request, so that any caller that does
not opt in keeps the legacy behaviour. The existing ``request_class``
hint (``"fg"`` / ``"bg"``) is still honoured — :class:`LookaheadRequestMeta`
is layered on top of it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

LOOKAHEAD_META_FIELDS = (
    "is_lookahead",
    "priority_class",
    "lookahead_group_id",
    "commit_deadline_ms",
    "slo_ttft_ms",
    "slo_tbt_ms",
    "arrival_time_ms",
)


@dataclasses.dataclass
class LookaheadRequestMeta:
    """Lookahead-aware metadata attached to a single request.

    Attributes:
        is_lookahead: ``True`` for a best-effort lookahead transform
            that may be deferred or rejected when LC requests need slack.
        priority_class: Coarse priority bucket (``"lc"`` or ``"be"``).
            Mapped onto :class:`RequestChannel` on the scheduler side
            (``"lc"`` → FOREGROUND, ``"be"`` → BACKGROUND).
        lookahead_group_id: Logical id linking a lookahead transform
            with its eventual commit, used for prefix-cache reuse.
        commit_deadline_ms: Wall-clock millisecond deadline by which
            the corresponding commit is expected.
        slo_ttft_ms: Per-request TTFT SLO (overrides the global
            ``LookaheadConfig.slo_target_ms`` when set).
        slo_tbt_ms: Per-request TBT SLO.
        arrival_time_ms: Optional client-side arrival timestamp; used
            by ``alg:schedule-prefill`` to compute slack.
    """

    is_lookahead: bool = False
    priority_class: str = "lc"
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
        if self.is_lookahead and self.priority_class != "be":
            raise ValueError(
                "LookaheadRequestMeta with is_lookahead=True must use priority_class='be'"
            )

    @property
    def is_be(self) -> bool:
        return self.priority_class == "be" or self.is_lookahead

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "is_lookahead": bool(self.is_lookahead),
            "priority_class": self.priority_class,
        }
        for name in (
            "lookahead_group_id",
            "commit_deadline_ms",
            "slo_ttft_ms",
            "slo_tbt_ms",
            "arrival_time_ms",
        ):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out


def parse_extra_body(
    extra_body: Optional[Dict[str, Any]],
    *,
    fallback_request_class: Optional[str] = None,
) -> LookaheadRequestMeta:
    """Parse a JSON-friendly ``extra_body`` dict into a metadata object.

    ``fallback_request_class`` may be used to honour the legacy
    top-level ``request_class`` ("fg"/"bg") field on the request, so
    that callers that have not migrated to the SmoothAgent ``extra_body``
    keep working.
    """
    body = dict(extra_body or {})

    is_lookahead = bool(body.get("is_lookahead", False))
    priority_class = body.get("priority_class")
    if priority_class is None:
        if is_lookahead:
            priority_class = "be"
        elif fallback_request_class == "bg":
            priority_class = "be"
        else:
            priority_class = "lc"

    return LookaheadRequestMeta(
        is_lookahead=is_lookahead,
        priority_class=str(priority_class),
        lookahead_group_id=_optional_str(body.get("lookahead_group_id")),
        commit_deadline_ms=_optional_float(body.get("commit_deadline_ms")),
        slo_ttft_ms=_optional_float(body.get("slo_ttft_ms")),
        slo_tbt_ms=_optional_float(body.get("slo_tbt_ms")),
        arrival_time_ms=_optional_float(body.get("arrival_time_ms")),
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
