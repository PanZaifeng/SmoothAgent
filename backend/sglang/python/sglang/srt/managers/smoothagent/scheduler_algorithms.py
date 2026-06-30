"""Reference implementations of the paper scheduling algorithms.

The functions here are **side-effect-free helpers** that take Python
data classes (``DecodeRequest`` / ``PrefillChunk`` / ``LCRequest``) and
return ``AdmissionDecision`` objects. They are intended for:

1. Symbolic verification: a unit-test harness can compare the production
   ``maybe_append_bg_prefill`` path against these reference helpers to
   prove the policies are equivalent.
2. New-feature scaffolding: integrators who want the paper-aligned policy
   verbatim — e.g., a PD-disaggregated prefill instance — can call the
   helpers directly without re-implementing the slack arithmetic.

Algorithm references: ``alg:schedule-prefill`` and ``alg:schedule-hybrid`` in
Section ``sec:colocate``.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, List, Optional, Sequence, Tuple

from sglang.srt.managers.smoothagent.batch_latency_estimator import (
    BatchLatencyEstimator,
    DecodeRequest,
    PrefillChunk,
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LCRequest:
    """Latency-critical prefill request as seen by ``alg:schedule-prefill``.

    Attributes:
        chunks: Ordered prefill chunks for this request.
        slo_ttft_ms: Per-request TTFT SLO.
        arrival_time_ms: Wall-clock arrival timestamp.
        rid: Optional request id, only used for diagnostics.
    """

    chunks: Sequence[PrefillChunk]
    slo_ttft_ms: float
    arrival_time_ms: float
    rid: Optional[str] = None


@dataclasses.dataclass
class AdmissionDecision:
    """Result of running one of the scheduler algorithms."""

    batch_decodes: List[DecodeRequest] = dataclasses.field(default_factory=list)
    batch_lc_chunks: List[PrefillChunk] = dataclasses.field(default_factory=list)
    batch_be_chunks: List[PrefillChunk] = dataclasses.field(default_factory=list)
    estimated_latency_ms: float = 0.0
    t_budget_ms: Optional[float] = None
    rejected_be: List[Tuple[PrefillChunk, str]] = dataclasses.field(default_factory=list)
    rejected_lc: List[Tuple[PrefillChunk, str]] = dataclasses.field(default_factory=list)
    rejected_decode: List[Tuple[DecodeRequest, str]] = dataclasses.field(default_factory=list)


# Type alias used by the disaggregated path. Returns the predicted prefill
# latency for an ``LCRequest``.
PrefillLatencyEstimator = Callable[[LCRequest], float]


# ---------------------------------------------------------------------------
# ``alg:schedule-hybrid`` — PD co-located
# ---------------------------------------------------------------------------


def schedule_hybrid_colocated(
    decode_queue: Sequence[DecodeRequest],
    lc_prefill_queue: Sequence[PrefillChunk],
    be_prefill_queue: Sequence[PrefillChunk],
    *,
    tbt_slo_ms: float,
    estimator: BatchLatencyEstimator,
) -> AdmissionDecision:
    """Implement ``alg:schedule-hybrid``.

    The function constructs the batch in priority order — decode first,
    then LC prefill, then BE prefill — and inserts a candidate only if
    the resulting predicted batch latency is within ``tbt_slo_ms``.
    """
    decision = AdmissionDecision(t_budget_ms=tbt_slo_ms)

    def _current_latency() -> float:
        return estimator.estimate(
            decision.batch_decodes,
            decision.batch_lc_chunks + decision.batch_be_chunks,
        )

    # Phase 1: decode requests.
    for req in decode_queue:
        decision.batch_decodes.append(req)
        if _current_latency() > tbt_slo_ms:
            decision.batch_decodes.pop()
            decision.rejected_decode.append((req, "exceeds TBT bound"))
            break

    # Phase 2: LC prefill chunks.
    for chunk in lc_prefill_queue:
        decision.batch_lc_chunks.append(chunk)
        if _current_latency() > tbt_slo_ms:
            decision.batch_lc_chunks.pop()
            decision.rejected_lc.append((chunk, "exceeds TBT bound"))
            break

    # Phase 3: BE prefill chunks.
    for chunk in be_prefill_queue:
        decision.batch_be_chunks.append(chunk)
        if _current_latency() > tbt_slo_ms:
            decision.batch_be_chunks.pop()
            decision.rejected_be.append((chunk, "exceeds TBT bound"))
            break

    decision.estimated_latency_ms = _current_latency()
    return decision


# ---------------------------------------------------------------------------
# ``alg:schedule-prefill`` — PD disaggregated
# ---------------------------------------------------------------------------


def schedule_prefill_disaggregated(
    lc_queue: Sequence[LCRequest],
    be_prefill_queue: Sequence[PrefillChunk],
    *,
    now_ms: float,
    estimator: BatchLatencyEstimator,
    prefill_latency_estimator: Optional[PrefillLatencyEstimator] = None,
) -> AdmissionDecision:
    """Implement ``alg:schedule-prefill``.

    ``prefill_latency_estimator`` is invoked once per LC request to
    estimate its prefill cost. If left as ``None``, the estimator
    defaults to ``estimator.estimate(decodes=(), prefill_chunks=lc.chunks)``,
    which is the natural choice when the prefill instance is otherwise
    idle.
    """
    decision = AdmissionDecision()

    if prefill_latency_estimator is None:

        def prefill_latency_estimator(req: LCRequest) -> float:  # type: ignore[misc]
            return estimator.estimate(decodes=(), prefill_chunks=req.chunks)

    # Compute cumulative LC prefill latency and minimum slack.
    cumulative = 0.0
    s_min = float("inf")
    for req in lc_queue:
        cumulative += float(prefill_latency_estimator(req))
        deadline = req.arrival_time_ms + req.slo_ttft_ms
        slack = deadline - now_ms - cumulative
        if slack < s_min:
            s_min = slack
    t_budget = max(0.0, s_min) if s_min != float("inf") else float("inf")
    decision.t_budget_ms = t_budget

    # The "next chunk" for each LC request is the head of its chunk list.
    decision.batch_lc_chunks = [req.chunks[0] for req in lc_queue if req.chunks]

    def _current_latency() -> float:
        return estimator.estimate(
            decodes=(),
            prefill_chunks=decision.batch_lc_chunks + decision.batch_be_chunks,
        )

    if t_budget == 0.0:
        decision.rejected_be.extend((c, "t_budget=0") for c in be_prefill_queue)
        decision.estimated_latency_ms = _current_latency()
        return decision

    for chunk in be_prefill_queue:
        decision.batch_be_chunks.append(chunk)
        if _current_latency() > t_budget:
            decision.batch_be_chunks.pop()
            decision.rejected_be.append((chunk, "exceeds t_budget"))
            break

    decision.estimated_latency_ms = _current_latency()
    return decision
