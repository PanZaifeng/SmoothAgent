"""Admission controller for SmoothAgent BE requests.

This module wires the standalone helpers in :mod:`scheduler_algorithms`
and :mod:`batch_latency_estimator` into a small controller that the
production scheduler can consult **as an optional alternative** to the
legacy ``SLOBudgetController.decide()`` path.

The controller is **opt-in**: it is activated by setting the environment
variable ``SGLANG_SMOOTHAGENT_PAPER_MODE=1`` (or by passing
``paper_mode_enabled=True`` directly when constructing it).

Why a separate class?
---------------------

``scheduler_lookahead_mixin.gate_bg_request`` already implements a
fully-functional admission policy. Re-implementing the paper algorithms
inside that mixin would either duplicate logic or replace a battle-tested
path. We instead provide a thin, testable adapter that:

1. Accepts the same inputs the mixin already has (a ``Req``, the current
   batch composition, KV pressure stats).
2. Returns ``(admit, reason, predicted_latency_ms)`` — a triple the mixin
   can fold into its existing logging and statistics.
3. Falls back to the legacy controller when the request does not carry
   SmoothAgent SLO fields, so enabling the flag at runtime is safe even
   for clients that have not migrated.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Iterable, Optional, Sequence, Tuple

from sglang.srt.managers.smoothagent.batch_latency_estimator import (
    BatchLatencyEstimator,
    BatchLatencyEstimatorConfig,
    DecodeRequest,
    PrefillChunk,
)
from sglang.srt.managers.smoothagent.req_field_pipeline import req_meta


PAPER_MODE_ENV_VAR = "SGLANG_SMOOTHAGENT_PAPER_MODE"


def paper_mode_enabled_default() -> bool:
    raw = os.environ.get(PAPER_MODE_ENV_VAR, "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclasses.dataclass
class AdmissionVerdict:
    """Result returned by :meth:`SmoothAgentAdmissionController.should_admit_be`."""

    admit: bool
    reason: str
    predicted_latency_ms: float = 0.0
    t_budget_ms: Optional[float] = None


@dataclasses.dataclass
class BatchSnapshot:
    """A scheduler-agnostic view of the batch composition.

    Production callers populate this from their existing data structures
    just before they evaluate a candidate BE chunk.

    Attributes:
        decodes: Decode requests already in the running batch.
        lc_chunks: LC prefill chunks already accepted into the next batch.
        be_chunks: BE prefill chunks already accepted into the next batch.
        waiting_queue_len: Number of LC requests waiting upstream of the
            current batch. The admission controller tightens the effective
            TBT bound when this number is large — backlog pressure is a
            stronger signal than per-batch latency alone in practice.
    """

    decodes: Sequence[DecodeRequest] = ()
    lc_chunks: Sequence[PrefillChunk] = ()
    be_chunks: Sequence[PrefillChunk] = ()
    waiting_queue_len: int = 0


class SmoothAgentAdmissionController:
    """SLO-aware admission decisions for the paper scheduler policies.

    On top of the strict batch-latency check, the controller also applies
    a **backlog penalty**: when the LC waiting queue is long, the effective
    TBT bound is shrunk so BE chunks are rejected more aggressively. This
    matches the paper's intent to trade BE throughput for LC tail latency
    under contention.
    """

    def __init__(
        self,
        *,
        global_slo_target_ms: float,
        estimator: Optional[BatchLatencyEstimator] = None,
        paper_mode_enabled: Optional[bool] = None,
        backlog_soft_threshold: int = 4,
        backlog_hard_threshold: int = 12,
    ) -> None:
        self.global_slo_target_ms = float(global_slo_target_ms)
        self.estimator = estimator or BatchLatencyEstimator(
            BatchLatencyEstimatorConfig()
        )
        self.paper_mode_enabled = (
            paper_mode_enabled
            if paper_mode_enabled is not None
            else paper_mode_enabled_default()
        )
        self.backlog_soft_threshold = int(backlog_soft_threshold)
        self.backlog_hard_threshold = int(backlog_hard_threshold)

    # ------------------------------------------------------------------
    # Per-request SLO helpers
    # ------------------------------------------------------------------

    def request_slo_ttft_ms(self, req: Any) -> float:
        meta = req_meta(req)
        if meta.slo_ttft_ms is not None:
            return float(meta.slo_ttft_ms)
        return self.global_slo_target_ms

    def request_slo_tbt_ms(self, req: Any) -> Optional[float]:
        meta = req_meta(req)
        if meta.slo_tbt_ms is not None:
            return float(meta.slo_tbt_ms)
        return None

    # ------------------------------------------------------------------
    # Paper-aligned admission decision
    # ------------------------------------------------------------------

    def should_admit_be(
        self,
        req: Any,
        candidate_chunk: PrefillChunk,
        snapshot: BatchSnapshot,
        *,
        mode: str = "colocated",
        tbt_slo_ms: Optional[float] = None,
        t_budget_ms: Optional[float] = None,
    ) -> AdmissionVerdict:
        """Decide whether to admit ``candidate_chunk`` into the batch.

        ``mode``:
            ``"colocated"`` follows ``alg:schedule-hybrid`` (TBT bound δ).
            Provide ``tbt_slo_ms`` to override the controller's default.
            ``"disaggregated"`` follows ``alg:schedule-prefill``;
            ``t_budget_ms`` must be supplied — typically the controller's
            caller has already computed it from the LC queue.
        """
        if not self.paper_mode_enabled:
            return AdmissionVerdict(
                admit=True,
                reason="paper-mode disabled",
                predicted_latency_ms=0.0,
            )

        if mode == "colocated":
            bound = (
                tbt_slo_ms
                if tbt_slo_ms is not None
                else (self.request_slo_tbt_ms(req) or self.global_slo_target_ms)
            )
            return self._decide_colocated(snapshot, candidate_chunk, bound)
        if mode == "disaggregated":
            if t_budget_ms is None:
                raise ValueError("disaggregated mode requires t_budget_ms")
            return self._decide_disaggregated(snapshot, candidate_chunk, t_budget_ms)
        raise ValueError(f"unknown admission mode: {mode!r}")

    # ------------------------------------------------------------------
    # Disaggregated t_budget computation from ``alg:schedule-prefill``.
    # ------------------------------------------------------------------

    def compute_t_budget_ms(
        self,
        lc_reqs: Iterable[Any],
        *,
        now_ms: float,
        prefill_latency_ms: "PrefillLatencyFn | None" = None,
    ) -> float:
        """Compute ``t_budget = max(0, min slack)`` over the LC queue."""
        cumulative = 0.0
        s_min = float("inf")
        any_lc = False
        for req in lc_reqs:
            any_lc = True
            cumulative += float(
                prefill_latency_ms(req) if prefill_latency_ms else 0.0
            )
            arrival = (
                req_meta(req).arrival_time_ms or 0.0
            )
            slack = arrival + self.request_slo_ttft_ms(req) - now_ms - cumulative
            if slack < s_min:
                s_min = slack
        if not any_lc:
            return float("inf")
        return max(0.0, s_min)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decide_colocated(
        self,
        snapshot: BatchSnapshot,
        candidate: PrefillChunk,
        tbt_slo_ms: float,
    ) -> AdmissionVerdict:
        # Backlog penalty: the deeper the LC queue, the smaller the
        # effective TBT bound we're willing to give to BE work. Above
        # the hard threshold we deny outright; between soft and hard we
        # interpolate the bound linearly down to half.
        q_len = max(0, int(snapshot.waiting_queue_len))
        effective_bound = float(tbt_slo_ms)
        if q_len >= self.backlog_hard_threshold:
            return AdmissionVerdict(
                admit=False,
                reason=f"backlog hard threshold (queue_len={q_len})",
                predicted_latency_ms=0.0,
                t_budget_ms=effective_bound,
            )
        if q_len > self.backlog_soft_threshold:
            span = max(
                1, self.backlog_hard_threshold - self.backlog_soft_threshold
            )
            ratio = (q_len - self.backlog_soft_threshold) / span
            effective_bound = tbt_slo_ms * (1.0 - 0.5 * ratio)

        new_be = list(snapshot.be_chunks) + [candidate]
        predicted = self.estimator.estimate(
            decodes=snapshot.decodes,
            prefill_chunks=list(snapshot.lc_chunks) + new_be,
        )
        if predicted > effective_bound:
            return AdmissionVerdict(
                admit=False,
                reason=(
                    f"exceeds TBT bound (effective={effective_bound:.1f}ms,"
                    f" backlog={q_len})"
                ),
                predicted_latency_ms=predicted,
                t_budget_ms=effective_bound,
            )
        return AdmissionVerdict(
            admit=True,
            reason="within TBT bound",
            predicted_latency_ms=predicted,
            t_budget_ms=effective_bound,
        )

    def _decide_disaggregated(
        self,
        snapshot: BatchSnapshot,
        candidate: PrefillChunk,
        t_budget_ms: float,
    ) -> AdmissionVerdict:
        if t_budget_ms <= 0.0:
            return AdmissionVerdict(
                admit=False,
                reason="t_budget=0",
                predicted_latency_ms=0.0,
                t_budget_ms=t_budget_ms,
            )
        new_be = list(snapshot.be_chunks) + [candidate]
        predicted = self.estimator.estimate(
            decodes=(),
            prefill_chunks=list(snapshot.lc_chunks) + new_be,
        )
        if predicted > t_budget_ms:
            return AdmissionVerdict(
                admit=False,
                reason="exceeds t_budget",
                predicted_latency_ms=predicted,
                t_budget_ms=t_budget_ms,
            )
        return AdmissionVerdict(
            admit=True,
            reason="within t_budget",
            predicted_latency_ms=predicted,
            t_budget_ms=t_budget_ms,
        )


# Type alias: callable producing prefill latency for an LC request.
from typing import Callable  # noqa: E402

PrefillLatencyFn = Callable[[Any], float]
