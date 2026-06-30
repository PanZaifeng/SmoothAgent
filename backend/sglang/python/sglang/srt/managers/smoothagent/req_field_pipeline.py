"""Helpers that copy SmoothAgent metadata onto a :class:`Req` instance.

The fields are added in three places (``GenerateReqInput`` in
``io_struct``, ``ChatCompletionRequest`` in ``protocol``, and
``TokenizedGenerateReqInput`` in ``io_struct``). The scheduler is the
final consumer — once a :class:`Req` is constructed it must mirror those
fields onto its ``smoothagent_*`` slots so that the new admission helpers
can read them without depending on the io-struct shape.

The functions below are split out of ``scheduler.py`` so that they can be
unit-tested without spinning up the full scheduler.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from sglang.srt.managers.smoothagent.smoothagent_meta import (
    LookaheadRequestMeta,
    parse_extra_body,
)


def attach_smoothagent_fields(req: Any, source: Any) -> None:
    """Copy ``source.{is_lookahead, priority_class, ...}`` to ``req.smoothagent_*``.

    ``source`` is typically a :class:`TokenizedGenerateReqInput`. The
    scheduler calls this once when it constructs ``req`` and once more
    when ``req`` is rehydrated from a session, so it must be idempotent.
    """
    req.smoothagent_is_lookahead = _opt_bool(getattr(source, "is_lookahead", None))
    req.smoothagent_priority_class = _opt_priority(
        getattr(source, "priority_class", None),
        is_lookahead=req.smoothagent_is_lookahead,
        request_class_hint=getattr(source, "request_class", None),
    )
    req.smoothagent_lookahead_group_id = _opt_str(
        getattr(source, "lookahead_group_id", None)
    )
    req.smoothagent_commit_deadline_ms = _opt_float(
        getattr(source, "commit_deadline_ms", None)
    )
    req.smoothagent_slo_ttft_ms = _opt_float(getattr(source, "slo_ttft_ms", None))
    req.smoothagent_slo_tbt_ms = _opt_float(getattr(source, "slo_tbt_ms", None))
    arrival_time_ms = _opt_float(getattr(source, "arrival_time_ms", None))
    if arrival_time_ms is not None:
        req.smoothagent_arrival_time_ms = arrival_time_ms
    elif getattr(req, "smoothagent_arrival_time_ms", None) is None:
        req.smoothagent_arrival_time_ms = time.time() * 1000.0


def req_meta(req: Any) -> LookaheadRequestMeta:
    """Return a :class:`LookaheadRequestMeta` view of ``req`` for the scheduler.

    Falls back to the legacy ``_request_class_hint`` ("fg" / "bg") when the
    SmoothAgent fields are absent, so callers can rely on a single shape.
    """
    if getattr(req, "smoothagent_is_lookahead", None) is None and getattr(
        req, "smoothagent_priority_class", None
    ) is None:
        return parse_extra_body(
            None,
            fallback_request_class=getattr(req, "_request_class_hint", None),
        )
    return LookaheadRequestMeta(
        is_lookahead=bool(getattr(req, "smoothagent_is_lookahead", False)),
        priority_class=str(getattr(req, "smoothagent_priority_class", "lc") or "lc"),
        lookahead_group_id=getattr(req, "smoothagent_lookahead_group_id", None),
        commit_deadline_ms=getattr(req, "smoothagent_commit_deadline_ms", None),
        slo_ttft_ms=getattr(req, "smoothagent_slo_ttft_ms", None),
        slo_tbt_ms=getattr(req, "smoothagent_slo_tbt_ms", None),
        arrival_time_ms=getattr(req, "smoothagent_arrival_time_ms", None),
    )


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _opt_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    return bool(v)


def _opt_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    return str(v)


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_priority(
    raw: Any,
    *,
    is_lookahead: Optional[bool],
    request_class_hint: Any,
) -> Optional[str]:
    if raw is not None:
        value = str(raw).lower()
        if value in ("lc", "be"):
            return value
    if is_lookahead is True:
        return "be"
    if request_class_hint == "bg":
        return "be"
    if is_lookahead is False or request_class_hint == "fg":
        return "lc"
    return None
