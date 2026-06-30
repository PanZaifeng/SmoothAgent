"""
SchedulerLookaheadMixin — Full BG scheduling integration for SGLang Scheduler.

Implements HyGen-style dual-queue, two-phase scheduling with all 8 controllers:
  Phase-Online: FG only (existing logic, untouched)
  Phase-Offline: if budget > 0, append BG chunked prefill into the same tick's batch

Also gates BG decode, tracks per-request token stats, and attaches ReduceSignals.
"""

import hashlib
import json
import logging
import math
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.managers.smoothagent.admission_controller import (
    BatchSnapshot,
    SmoothAgentAdmissionController,
    paper_mode_enabled_default,
)
from sglang.srt.managers.smoothagent.batch_latency_estimator import (
    BatchLatencyEstimator as _BLEstimator,
    BatchLatencyEstimatorConfig as _BLEstimatorConfig,
    DecodeRequest as _BLDecodeRequest,
    PrefillChunk as _BLPrefillChunk,
    default_gemm_table_qwen3_8b_h100 as _bl_default_gemm_qwen3_8b_h100,
)
from sglang.srt.managers.smoothagent.snapshot_builder import (
    build_paper_batch_snapshot as _bl_build_snapshot,
    paper_estimate_next_chunk_tokens as _bl_estimate_chunk,
    paper_estimate_prefix_kv_len as _bl_estimate_prefix,
    paper_remaining_prefill_tokens as _bl_remaining_prefill,
)
from sglang.srt.managers.smoothagent.marginal_model import (
    closed_form_marginal_ms as _bl_closed_form_marginal_ms,
    eq1_marginal_ms as _bl_eq1_marginal_ms,
)
from sglang.srt.managers.io_struct import (
    LookaheadControlReqInput,
    LookaheadControlReqOutput,
)
from sglang.srt.managers.lookahead_config import LookaheadConfig
from sglang.srt.managers.lookahead_context_engine import (
    LookaheadMainState,
    LookaheadState,
    should_commit,
    should_trigger,
    transform,
)
from sglang.srt.managers.lookahead_controller import (
    AntiStarvationGuard,
    BGDecision,
    BGSchedulingStats,
    ChunkMixer,
    DriftDetector,
    ExtremeBurstGuard,
    LatencyPredictor,
    ReduceSignalLevel,
    SLOBudgetController,
    SoftHardDetector,
)
from sglang.srt.managers.schedule_batch import (
    LookaheadResult,
    ReduceRequestType,
    Req,
    RequestChannel,
)
from sglang.srt.sampling.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class SchedulerLookaheadMixin:
    """Mixed into Scheduler via multiple inheritance."""

    # ── Initialization ───────────────────────────────────────────────

    def init_lookahead_scheduling(self):
        """Called from Scheduler.__init__ when --enable-bg-scheduling is set."""
        cfg = LookaheadConfig.from_server_args(self.server_args)
        self._lookahead_config = cfg

        # Core controllers
        self._latency_predictor = LatencyPredictor(
            alpha=cfg.ema_alpha,
            burst_multiplier=cfg.burst_spike_multiplier,
        )
        self._slo_budget_ctrl = SLOBudgetController(cfg)
        self._chunk_mixer = ChunkMixer(cfg)
        self._soft_hard_detector = SoftHardDetector(cfg)
        self._anti_starvation = AntiStarvationGuard(cfg)
        self._drift_detector = DriftDetector(cfg)
        self._burst_guard = ExtremeBurstGuard(cfg)
        self._bg_stats = BGSchedulingStats()

        # ── SmoothAgent paper-aligned admission controller (opt-in) ───────
        # When SGLANG_SMOOTHAGENT_PAPER_MODE=1, BG admission also consults a
        # controller built around BatchLatencyEstimator
        # (``eq:latency-model``). The legacy SLOBudgetController remains the
        # fallback baseline. Both run side-by-side: a request is admitted only
        # when **both** controllers agree.
        self._smoothagent_paper_mode: bool = paper_mode_enabled_default()
        # Use the calibrated Qwen3-8B/H100 GEMM table by default; production
        # deployments on different model/GPU combinations should override
        # via a future server arg or env var. Falls back to defaults if any
        # calibration step fails.
        try:
            estimator = _BLEstimator(
                _BLEstimatorConfig(gemm_table=_bl_default_gemm_qwen3_8b_h100())
            )
        except Exception:
            estimator = _BLEstimator()
        self._smoothagent_admission = SmoothAgentAdmissionController(
            global_slo_target_ms=cfg.slo_target_ms,
            estimator=estimator,
            paper_mode_enabled=self._smoothagent_paper_mode,
        )
        if self._smoothagent_paper_mode:
            logger.info(
                "SmoothAgent paper-mode admission ENABLED "
                "(eq:latency-model + scheduler algorithms). Reqs without "
                "explicit slo_ttft_ms "
                "fall back to global slo_target_ms=%.0fms.",
                cfg.slo_target_ms,
            )

        # Paper-mode denial accounting (kept on the mixin so we don't grow
        # BGSchedulingStats — the existing dataclass is a shared contract).
        self._smoothagent_paper_denied_total: int = 0
        self._smoothagent_paper_admitted_total: int = 0
        self._smoothagent_last_predicted_latency_ms: float = 0.0
        self._smoothagent_last_t_budget_ms: float = 0.0
        # Colocated ``alg:schedule-hybrid`` LC-prefill δ gate accounting (additive —
        # only meaningful when SGLANG_SMOOTHAGENT_COLO_TBT_GATE_LC=1).
        self._smoothagent_lc_prefill_admitted_total: int = 0
        self._smoothagent_lc_prefill_deferred_total: int = 0
        # SGLANG_SMOOTHAGENT_COLO_TBT_LC_GATE_PASSIVE=1 mode: count LC-chunk
        # admissions that would have been deferred but were force-admitted
        # so the BE gate can still see lc_marginal without LC prefill being
        # gated. Stays 0 unless GATE_LC=1 AND LC_GATE_PASSIVE=1.
        self._smoothagent_lc_prefill_passive_skips_total: int = 0
        # Per-tick LC-prefill gate state — reset every prefill-batch build
        # by :meth:`_colo_lc_gate_begin_tick`. No cross-tick ledger.
        self._colo_tbt_lc_marginal_ms: float = 0.0
        self._colo_tbt_decode_baseline_cached_ms: float = 0.0

        # Marginal-cost calibration coefficients used when
        # SGLANG_SMOOTHAGENT_MARGINAL_MODE=closed_form. Defaults match the
        # ``linear_attn`` model fit on N_lc≥2 data — see
        # ``runs/calibration_v2/COMPARISON.md`` (2026-05-12).
        #   marginal_ms = c0 + c_attn · (q · prefix + q·(q+1)/2)
        self._marginal_c0_ms: float = 1.25
        self._marginal_c_attn: float = 1.93e-6
        self._hybrid_gamma_launch_ms: float = 0.0
        self._hybrid_alpha_p_0: Optional[float] = None
        self._hybrid_kappa: float = 0.0
        self._load_marginal_coeffs_from_env_or_file()

        # BG queue and chunked state. ``bg_chunked_reqs`` (paper-mode
        # multi-chunked) holds every in-flight BG chunked req; the legacy
        # ``bg_chunked_req`` accessor below maps to the first element.
        self.bg_waiting_queue: List[Req] = []
        self.bg_chunked_reqs: List[Req] = []
        self.promoted_lookahead_req: Optional[Req] = None
        self._bg_decode_paused_reqs: List[Req] = []

        # Interleave counter: increments on every FG decode tick that runs,
        # resets whenever any prefill batch fires. Used in scheduler.py to
        # decide when to steal a tick from FG decode for BG-only prefill.
        # Only consulted when is_mixed_chunk is False (non-mixed fallback);
        # with mixed chunking every BG-only prefill tick already carries the
        # running_batch decode reqs in the same forward pass, so no tick
        # stealing is needed.
        self._decode_ticks_since_bg_prefill: int = 0

        # Lookahead's BG prefill path is designed to piggyback on FG decode
        # via mixed chunking (scheduler.py:2285): the BG-only prefill batch
        # built by maybe_append_bg_prefill absorbs running_batch decode reqs
        # into the same forward pass, so BG prefill fills the FFN capacity
        # that decode (memory-bound on KV reads) leaves idle. Without this,
        # BG prefill and FG decode serialize tick-by-tick, which defeats the
        # whole point of running BG during decode. Force-enable mixed_chunk
        # here; warn if the user explicitly disabled --enable-mixed-chunk so
        # the override is discoverable.
        #
        # Note: init_lookahead_scheduling runs BEFORE init_chunked_prefill
        # (see scheduler.py:364 → 741 → 367 ordering), so self.is_mixed_chunk
        # and self.chunked_prefill_size don't exist yet. We flip the source
        # of truth (server_args.enable_mixed_chunk) so that init_chunked_prefill
        # computes the right value when it runs moments later.
        chunked_prefill_size = self.server_args.chunked_prefill_size
        chunked_enabled = chunked_prefill_size is not None and chunked_prefill_size > 0
        if chunked_enabled and not self.server_args.enable_mixed_chunk:
            logger.warning(
                "BG scheduling enabled: forcing server_args.enable_mixed_chunk=True "
                "so BG prefill can piggyback on FG decode in a single forward pass. "
                "This overrides --enable-mixed-chunk=False."
            )
            self.server_args.enable_mixed_chunk = True

        logger.info(
            "BG scheduling initialized: slo_target=%.0fms, max_chunk=%d, "
            "bg_max_inflight=%d, fg_backlog_soft=%d, "
            "kv_pressure_deny=%.2f, anti_starvation=%.0fs",
            cfg.slo_target_ms,
            cfg.bg_max_chunk_tokens,
            cfg.bg_max_inflight_reqs,
            cfg.bg_admission_fg_backlog_threshold,
            cfg.bg_admission_kv_pressure_ratio,
            cfg.anti_starvation_max_wait_s,
        )

    # ── chunked_req backward-compat shim (singular slot view) ────────

    @property
    def bg_chunked_req(self) -> Optional[Req]:
        # Returns the first in-flight BG chunked req or None. Existing
        # callers that read ``self.bg_chunked_req`` continue working;
        # paper-mode multi-chunked code should iterate
        # ``self.bg_chunked_reqs`` directly.
        return self.bg_chunked_reqs[0] if self.bg_chunked_reqs else None

    @bg_chunked_req.setter
    def bg_chunked_req(self, val: Optional[Req]) -> None:
        # Singular-slot writer: setting None clears the list; setting a
        # Req replaces with one element.
        self.bg_chunked_reqs = [val] if val is not None else []

    # ── Admission gate ───────────────────────────────────────────────

    def gate_bg_request(self, req: Req) -> bool:
        """
        Admission gate for BG requests.
        Returns True if admitted (caller adds to normal waiting_queue),
        False if denied (caller returns 503).

        When SmoothAgent paper-mode is enabled, both the legacy
        ``SLOBudgetController`` and the paper-aligned admission controller
        must agree before a request is admitted.
        """
        free_blocks = self.token_to_kv_pool_allocator.available_size()
        total_blocks = self.token_to_kv_pool_allocator.size
        fg_queue_len = len(self.waiting_queue)

        admitted, reason = self._slo_budget_ctrl.should_admit_bg(
            fg_queue_len, free_blocks, total_blocks
        )

        if admitted and self._smoothagent_paper_mode:
            paper_admitted, paper_reason = self._gate_bg_request_paper_mode(req)
            if paper_admitted:
                self._smoothagent_paper_admitted_total += 1
            else:
                self._smoothagent_paper_denied_total += 1
                admitted = False
                reason = f"paper-mode: {paper_reason}"

        if not admitted:
            self._bg_stats.bg_denied_total += 1
            logger.debug("BG admission denied rid=%s: %s", req.rid, reason)
            return False

        self._bg_stats.bg_admitted_total += 1
        logger.debug("BG admitted rid=%s", req.rid)
        return True

    def _gate_bg_request_paper_mode(self, req: Req) -> tuple:
        """Run the SmoothAgent paper-aligned admission gate.

        Routes to ``alg:schedule-prefill`` when this scheduler instance is
        the prefill side of a disaggregated cluster, otherwise to
        ``alg:schedule-hybrid``.

        For colocated, the candidate chunk size mirrors what
        ``maybe_append_bg_prefill`` would actually issue this tick. For
        disaggregated, ``t_budget_ms`` is computed from the LC waiting
        queue's TTFT slack.
        """
        chunk_tokens = self._paper_estimate_next_chunk_tokens(req)
        prefix_kv_len = self._paper_estimate_prefix_kv_len(req)
        candidate = _BLPrefillChunk(
            chunk_tokens=max(1, chunk_tokens),
            prefix_kv_len=max(0, prefix_kv_len),
        )

        snapshot = self._build_paper_batch_snapshot()
        if self._is_paper_disaggregated_prefill():
            t_budget_ms = self._compute_paper_t_budget_ms()
            if math.isfinite(t_budget_ms):
                self._smoothagent_last_t_budget_ms = float(t_budget_ms)
            verdict = self._smoothagent_admission.should_admit_be(
                req,
                candidate,
                snapshot,
                mode="disaggregated",
                t_budget_ms=t_budget_ms,
            )
        else:
            verdict = self._smoothagent_admission.should_admit_be(
                req,
                candidate,
                snapshot,
                mode="colocated",
                tbt_slo_ms=self._smoothagent_admission.request_slo_tbt_ms(req),
            )
        self._smoothagent_last_predicted_latency_ms = float(
            verdict.predicted_latency_ms
        )
        if verdict.t_budget_ms is not None and math.isfinite(verdict.t_budget_ms):
            self._smoothagent_last_t_budget_ms = float(verdict.t_budget_ms)
        return verdict.admit, verdict.reason

    def _is_paper_disaggregated_prefill(self) -> bool:
        """True when the underlying scheduler is a PD-disaggregated prefill node."""
        mode = getattr(self, "disaggregation_mode", None)
        if mode is None:
            return False
        # Compare by ``.value`` so we don't have to import the enum here
        # (avoids a hard dep when the disaggregation module is missing).
        return getattr(mode, "value", None) == "prefill"

    def _paper_estimate_next_chunk_tokens(self, req: Req) -> int:
        """Forward to the standalone helper, with the mixer plumbed in."""
        try:
            tbt_ms = float(self._latency_predictor.predicted_tbt_ms or 0.0)
        except AttributeError:
            tbt_ms = 0.0
        return _bl_estimate_chunk(
            req,
            bg_max_chunk_tokens=int(self._lookahead_config.bg_max_chunk_tokens),
            predicted_tbt_ms=tbt_ms,
            chunk_mixer_compute=getattr(
                self._chunk_mixer, "compute_chunk_size", None
            ),
        )

    @staticmethod
    def _paper_estimate_prefix_kv_len(req: Req) -> int:
        return _bl_estimate_prefix(req)

    def _load_marginal_coeffs_from_env_or_file(self) -> None:
        """Populate marginal-cost calibration coefficients.

        Priority (highest first):
          1. ``SGLANG_SMOOTHAGENT_MARGINAL_C0_MS`` / ``..._C_ATTN`` env vars
          2. JSON file at ``SGLANG_SMOOTHAGENT_MARGINAL_COEFFS_JSON``, picking
             ``models.linear_attn.coefficients.{c0, c_attn}``.
          3. Hardcoded defaults from the 2026-05-12 N_lc≥2 fit
             (already set in ``__init__``).
        """
        # 1. Env scalars
        env_c0 = os.environ.get("SGLANG_SMOOTHAGENT_MARGINAL_C0_MS")
        env_c_attn = os.environ.get("SGLANG_SMOOTHAGENT_MARGINAL_C_ATTN")
        if env_c0 is not None:
            try:
                self._marginal_c0_ms = float(env_c0)
            except ValueError:
                pass
        if env_c_attn is not None:
            try:
                self._marginal_c_attn = float(env_c_attn)
            except ValueError:
                pass

        # 2. JSON file
        path = os.environ.get("SGLANG_SMOOTHAGENT_MARGINAL_COEFFS_JSON")
        if path and env_c0 is None and env_c_attn is None:
            try:
                import json as _json
                with open(path, "r", encoding="utf-8") as fh:
                    data = _json.load(fh)
                coeffs = data.get("models", {}).get("linear_attn", {}).get(
                    "coefficients", {}
                )
                if "c0" in coeffs:
                    self._marginal_c0_ms = float(coeffs["c0"])
                if "c_attn" in coeffs:
                    self._marginal_c_attn = float(coeffs["c_attn"])
            except (OSError, ValueError, KeyError, TypeError):
                # Silent fallback to embedded defaults.
                pass

        # Hybrid ``eq:latency-model`` extension used by
        # SGLANG_SMOOTHAGENT_MARGINAL_MODE=estimator.
        # Env scalars override JSON. ``alpha_p_0`` remains optional so the
        # estimator's configured alpha_p is still the backwards-compatible
        # fallback when no hybrid calibration is provided.
        env_gamma = os.environ.get("SGLANG_SMOOTHAGENT_HYBRID_GAMMA_LAUNCH_MS")
        env_alpha = os.environ.get("SGLANG_SMOOTHAGENT_HYBRID_ALPHA_P_0")
        env_kappa = os.environ.get("SGLANG_SMOOTHAGENT_HYBRID_KAPPA")
        if env_gamma is not None:
            try:
                self._hybrid_gamma_launch_ms = max(0.0, float(env_gamma))
            except ValueError:
                pass
        if env_alpha is not None:
            try:
                self._hybrid_alpha_p_0 = max(0.0, float(env_alpha))
            except ValueError:
                pass
        if env_kappa is not None:
            try:
                self._hybrid_kappa = max(0.0, float(env_kappa))
            except ValueError:
                pass

        hybrid_path = os.environ.get("SGLANG_SMOOTHAGENT_HYBRID_COEFFS_JSON")
        if hybrid_path:
            try:
                import json as _json
                with open(hybrid_path, "r", encoding="utf-8") as fh:
                    data = _json.load(fh)
                if env_gamma is None and "gamma_launch_ms" in data:
                    self._hybrid_gamma_launch_ms = max(
                        0.0, float(data["gamma_launch_ms"])
                    )
                if env_alpha is None and "alpha_p_0" in data:
                    self._hybrid_alpha_p_0 = max(0.0, float(data["alpha_p_0"]))
                if env_kappa is None and "kappa" in data:
                    self._hybrid_kappa = max(0.0, float(data["kappa"]))
            except (OSError, ValueError, KeyError, TypeError):
                pass

    def _paper_decode_forward_tokens(self) -> int:
        """``eq:latency-model`` ``M`` contribution from next-tick decodes."""
        running = getattr(self, "running_batch", None)
        total = 0
        for req in getattr(running, "reqs", []) or []:
            if req is None:
                continue
            if int(getattr(req, "extend_input_len", 0) or 0) > 0:
                continue
            kv_committed = int(getattr(req, "kv_committed_len", 0) or 0)
            output_len = len(getattr(req, "output_ids", []) or [])
            if kv_committed + output_len > 0:
                total += 1
        return total

    @staticmethod
    def _paper_next_prefill_forward_tokens(req: Req, chunk_cap: int) -> int:
        """Best-effort ``M`` contribution for req's next prefill chunk."""
        remaining = max(0, int(_bl_remaining_prefill(req)))
        if remaining <= 0:
            return 0
        ext = int(getattr(req, "extend_input_len", 0) or 0)
        if ext > 0:
            return max(1, min(ext, remaining, chunk_cap))
        return max(1, min(chunk_cap, remaining))

    @staticmethod
    def _smoothagent_slack_reserve_ms() -> float:
        try:
            return max(
                0.0,
                float(os.environ.get("SGLANG_SMOOTHAGENT_SLACK_RESERVE_MS", "0.0")),
            )
        except (TypeError, ValueError):
            return 0.0

    def _per_lc_slack_ms(
        self,
        lc_req: Req,
        extra_be_reqs: Sequence[Req] = (),
        in_flight_be_reqs: Optional[Sequence[Req]] = None,
        in_flight_be_cost_state: Optional[Dict[int, tuple]] = None,
    ) -> float:
        """Single per-tick SLO slack for one LC, fully recomputed.

        slack = SLO − elapsed − LC_remaining_prefill_ms − Σ(BE contention)

        BE contention = Σ_{BE ∈ in-flight ∪ extra} overlap_chunks · marginal · discount
            where overlap_chunks = min(LC_remaining_chunks, BE_remaining_chunks).

        in-flight BEs come from ``self.bg_chunked_reqs`` by default (paused ones
        skipped, since their KV is held but they won't run). ``extra_be_reqs``
        covers candidates already admitted this tick but not yet in
        ``bg_chunked_reqs``. ``in_flight_be_cost_state`` lets callers preserve
        pre-continuation remaining/prefix state after ``adder.add_chunked_req``
        mutates ``fill_ids`` for the same forward pass.

        Env knobs:
          - SGLANG_SMOOTHAGENT_LC_PREFILL_MS_PER_TOKEN (default 0.12)
          - SGLANG_SMOOTHAGENT_BE_MARGINAL_COST_MS    (default 5.0)
          - SGLANG_SMOOTHAGENT_PRECHARGE_DISCOUNT     (default 1.0; <1 ⇒ less pessimistic)

        Returns slack in ms (can be negative — caller decides admission /
        preemption based on threshold).
        """
        slo = float(self._smoothagent_admission.request_slo_ttft_ms(lc_req))
        now_ms = time.time() * 1000.0
        arrival_ms = float(getattr(lc_req, "smoothagent_arrival_time_ms", None) or now_ms)
        elapsed_ms = max(0.0, now_ms - arrival_ms)

        lc_remaining_tokens = int(_bl_remaining_prefill(lc_req))
        per_token_ms = float(
            os.environ.get("SGLANG_SMOOTHAGENT_LC_PREFILL_MS_PER_TOKEN", "0.12")
        )
        lc_remaining_prefill_ms = lc_remaining_tokens * per_token_ms

        chunk_cap = max(1, int(self._lookahead_config.bg_max_chunk_tokens))
        lc_remaining_chunks = max(0, (lc_remaining_tokens + chunk_cap - 1) // chunk_cap)

        discount = float(
            os.environ.get("SGLANG_SMOOTHAGENT_PRECHARGE_DISCOUNT", "1.0")
        )
        mode = os.environ.get("SGLANG_SMOOTHAGENT_MARGINAL_MODE", "flat").lower()
        flat_marginal = float(
            os.environ.get("SGLANG_SMOOTHAGENT_BE_MARGINAL_COST_MS", "5.0")
        )

        be_cost_ms = 0.0
        cost_state = in_flight_be_cost_state or {}

        def _be_remaining_tokens(be: Req) -> int:
            state = cost_state.get(id(be))
            if state is not None:
                return max(0, int(state[0]))
            return max(0, int(_bl_remaining_prefill(be)))

        def _be_prefix_tokens(be: Req) -> int:
            state = cost_state.get(id(be))
            if state is not None:
                return max(0, int(state[1]))
            return max(0, int(_bl_estimate_prefix(be)))

        def _be_next_forward_tokens(be: Req) -> int:
            remaining = _be_remaining_tokens(be)
            if remaining <= 0:
                return 0
            ext = int(getattr(be, "extend_input_len", 0) or 0)
            if ext > 0:
                return max(1, min(ext, remaining, chunk_cap))
            return max(1, min(chunk_cap, remaining))

        in_flight = (
            list(in_flight_be_reqs)
            if in_flight_be_reqs is not None
            else list(getattr(self, "bg_chunked_reqs", []) or [])
        )
        active_bes: List[Req] = []
        for be in list(in_flight) + list(extra_be_reqs):
            if be is None:
                continue
            # Paused BEs hold KV but don't run this tick → no contention.
            if getattr(be, "smoothagent_be_paused", False):
                continue
            if _be_remaining_tokens(be) <= 0:
                continue
            active_bes.append(be)

        if mode == "closed_form" and active_bes:
            c0 = float(self._marginal_c0_ms)
            c_attn = float(self._marginal_c_attn)
            for be in active_bes:
                be_remaining_tokens = _be_remaining_tokens(be)
                be_remaining_chunks = (
                    be_remaining_tokens + chunk_cap - 1
                ) // chunk_cap
                overlap = min(lc_remaining_chunks, be_remaining_chunks)
                q_be = min(chunk_cap, be_remaining_tokens)
                prefix_be = _be_prefix_tokens(be)
                per_tick = _bl_closed_form_marginal_ms(
                    q_be=q_be, prefix_be=prefix_be, c0=c0, c_attn=c_attn,
                )
                be_cost_ms += overlap * per_tick * discount
        elif mode == "estimator" and active_bes:
            est = getattr(self._smoothagent_admission, "estimator", None)
            default_alpha_p = (
                float(getattr(est.config, "alpha_p", 5e-6))
                if est is not None
                else 5e-6
            )
            alpha_p = (
                float(self._hybrid_alpha_p_0)
                if self._hybrid_alpha_p_0 is not None
                else default_alpha_p
            )
            gamma_launch_ms = float(self._hybrid_gamma_launch_ms)
            kappa = float(self._hybrid_kappa)
            gemm_cost_fn = (
                est.gemm_cost if est is not None else (lambda _m: 0.0)
            )
            lc_base_reqs = [
                r for r in (getattr(self, "chunked_reqs", []) or []) if r is not None
            ]
            if all(r is not lc_req for r in lc_base_reqs):
                lc_base_reqs.append(lc_req)
            lc_active_count = len(lc_base_reqs)
            lc_forward_tokens = sum(
                self._paper_next_prefill_forward_tokens(r, chunk_cap)
                for r in lc_base_reqs
            )
            decode_forward_tokens = self._paper_decode_forward_tokens()
            be_forward_tokens = [_be_next_forward_tokens(be) for be in active_bes]
            total_be_forward_tokens = sum(be_forward_tokens)
            for be, q_be in zip(active_bes, be_forward_tokens):
                be_remaining_tokens = _be_remaining_tokens(be)
                be_remaining_chunks = (
                    be_remaining_tokens + chunk_cap - 1
                ) // chunk_cap
                overlap = min(lc_remaining_chunks, be_remaining_chunks)
                prefix_be = _be_prefix_tokens(be)
                m_base = (
                    decode_forward_tokens
                    + lc_forward_tokens
                    + max(0, total_be_forward_tokens - q_be)
                )
                per_tick = _bl_eq1_marginal_ms(
                    q_be=q_be,
                    prefix_be=prefix_be,
                    m_base=m_base,
                    alpha_p=alpha_p,
                    gemm_cost=gemm_cost_fn,
                    n_other=max(0, lc_active_count + len(active_bes) - 1),
                    gamma_launch_ms=gamma_launch_ms,
                    kappa=kappa,
                )
                be_cost_ms += overlap * per_tick * discount
        else:  # flat (default — backwards-compatible)
            for be in active_bes:
                be_remaining_tokens = _be_remaining_tokens(be)
                be_remaining_chunks = (
                    be_remaining_tokens + chunk_cap - 1
                ) // chunk_cap
                overlap = min(lc_remaining_chunks, be_remaining_chunks)
                be_cost_ms += overlap * flat_marginal * discount

        return slo - elapsed_ms - lc_remaining_prefill_ms - be_cost_ms

    def _per_lc_slack_gate(
        self,
        candidate: Req,
        candidate_chunk,
        snapshot,
        in_flight_lcs: List[Req],
        pending_be_reqs: Sequence[Req] = (),
        in_flight_be_reqs: Optional[Sequence[Req]] = None,
        in_flight_be_cost_state: Optional[Dict[int, tuple]] = None,
    ):
        """Admit candidate iff every in-flight LC's slack stays above reserve.

        Pure per-tick decision — no persistent ledger. ``pending_be_reqs``
        covers candidates already admitted earlier in the same tick.

        Returns (admit, reason, predicted_latency_ms, marginal_ms).
        """
        estimator = self._smoothagent_admission.estimator
        be_chunks_with = list(snapshot.be_chunks) + [candidate_chunk]
        lc_chunks = list(snapshot.lc_chunks)
        pred_with = float(
            estimator.estimate(decodes=(), prefill_chunks=lc_chunks + be_chunks_with)
        )
        marginal = float(
            os.environ.get("SGLANG_SMOOTHAGENT_BE_MARGINAL_COST_MS", "5.0")
        )

        extra = list(pending_be_reqs) + [candidate]
        reserve_ms = self._smoothagent_slack_reserve_ms()
        for lc in in_flight_lcs:
            slack_after = self._per_lc_slack_ms(
                lc,
                extra_be_reqs=extra,
                in_flight_be_reqs=in_flight_be_reqs,
                in_flight_be_cost_state=in_flight_be_cost_state,
            )
            if slack_after < reserve_ms:
                return (
                    False,
                    f"slack<{reserve_ms:.0f} after admit "
                    f"(rid={lc.rid}, slack={slack_after:.0f}ms)",
                    pred_with,
                    marginal,
                )
        return True, "slack ok", pred_with, marginal

    @staticmethod
    def _colo_tbt_slo_ms() -> float:
        """TBT SLO bound δ (ms) for the ``alg:schedule-hybrid`` slack gate."""
        try:
            return max(
                1.0,
                float(os.environ.get("SGLANG_SMOOTHAGENT_COLO_TBT_SLO_MS", "50.0")),
            )
        except (TypeError, ValueError):
            return 50.0

    def _colo_tbt_slack_ms(
        self,
        extra_be_reqs: Sequence[Req] = (),
        in_flight_be_reqs: Optional[Sequence[Req]] = None,
        in_flight_be_cost_state: Optional[Dict[int, tuple]] = None,
        decode_baseline_ms: Optional[float] = None,
        lc_marginal_ms: Optional[float] = None,
    ) -> float:
        """Per-tick TBT slack for the ``alg:schedule-hybrid`` gate.

        slack = δ_TBT − baseline_tick_ms − Σ_BE marginal

        When the LC-prefill gate is active (``decode_baseline_ms`` supplied),
        ``baseline_tick_ms`` becomes the analytical decode-only cost plus the
        explicit per-tick LC-prefill marginal (``lc_marginal_ms``). This keeps
        LC prefill counted exactly once: the measured ``predicted_tbt_ms`` EMA
        already absorbs LC prefill, so it cannot be the baseline once LC
        marginals are added on top. When ``decode_baseline_ms`` is ``None``
        (default) the measured-EMA baseline is used — behaviour unchanged.

        * ``δ_TBT`` — TBT SLO bound (env SGLANG_SMOOTHAGENT_COLO_TBT_SLO_MS).
        * ``baseline_tick_ms`` — measured decode-tick latency from the
          LatencyPredictor EMA. Using the *measured* tick cost sidesteps
          the structurally under-predicting full-batch estimator
          (runs/calibration_v2/COMPARISON.md: refitting α_p only moves
          MAPE 74%→72%).
        * ``Σ_BE marginal`` — per-tick marginal cost of every active
          (non-paused) BE prefill chunk, via the calibrated marginal
          model (SGLANG_SMOOTHAGENT_MARGINAL_MODE — shared with the
          PD-disagg TTFT gate). Unlike :meth:`_per_lc_slack_ms` this is a
          single-tick cost: no overlap-chunks multiply, since TBT is a
          per-tick (not cumulative) bound.

        Returns slack in ms (negative ⇒ over budget ⇒ deny / preempt).
        """
        delta = self._colo_tbt_slo_ms()
        if decode_baseline_ms is not None:
            baseline_ms = float(decode_baseline_ms) + float(lc_marginal_ms or 0.0)
        else:
            try:
                baseline_ms = float(self._latency_predictor.predicted_tbt_ms or 0.0)
            except AttributeError:
                baseline_ms = 0.0

        chunk_cap = max(1, int(self._lookahead_config.bg_max_chunk_tokens))
        discount = float(
            os.environ.get("SGLANG_SMOOTHAGENT_PRECHARGE_DISCOUNT", "1.0")
        )
        mode = os.environ.get("SGLANG_SMOOTHAGENT_MARGINAL_MODE", "flat").lower()
        flat_marginal = float(
            os.environ.get("SGLANG_SMOOTHAGENT_BE_MARGINAL_COST_MS", "5.0")
        )
        cost_state = in_flight_be_cost_state or {}

        def _remaining(be: Req) -> int:
            state = cost_state.get(id(be))
            if state is not None:
                return max(0, int(state[0]))
            return max(0, int(_bl_remaining_prefill(be)))

        def _prefix(be: Req) -> int:
            state = cost_state.get(id(be))
            if state is not None:
                return max(0, int(state[1]))
            return max(0, int(_bl_estimate_prefix(be)))

        in_flight = (
            list(in_flight_be_reqs)
            if in_flight_be_reqs is not None
            else list(getattr(self, "bg_chunked_reqs", []) or [])
        )
        active_bes: List[Req] = []
        for be in list(in_flight) + list(extra_be_reqs):
            if be is None or getattr(be, "smoothagent_be_paused", False):
                continue
            if _remaining(be) <= 0:
                continue
            active_bes.append(be)

        be_cost_ms = 0.0
        if mode == "closed_form" and active_bes:
            c0 = float(self._marginal_c0_ms)
            c_attn = float(self._marginal_c_attn)
            for be in active_bes:
                be_cost_ms += discount * _bl_closed_form_marginal_ms(
                    q_be=min(chunk_cap, _remaining(be)),
                    prefix_be=_prefix(be),
                    c0=c0,
                    c_attn=c_attn,
                )
        elif mode == "estimator" and active_bes:
            est = getattr(self._smoothagent_admission, "estimator", None)
            default_alpha_p = (
                float(getattr(est.config, "alpha_p", 5e-6))
                if est is not None
                else 5e-6
            )
            alpha_p = (
                float(self._hybrid_alpha_p_0)
                if self._hybrid_alpha_p_0 is not None
                else default_alpha_p
            )
            gemm_cost_fn = est.gemm_cost if est is not None else (lambda _m: 0.0)
            lc_reqs = [
                r
                for r in (getattr(self, "chunked_reqs", []) or [])
                if r is not None
            ]
            decode_forward_tokens = self._paper_decode_forward_tokens()
            lc_forward_tokens = sum(
                self._paper_next_prefill_forward_tokens(r, chunk_cap)
                for r in lc_reqs
            )
            be_forward = [
                max(1, min(chunk_cap, _remaining(be))) for be in active_bes
            ]
            total_be_forward = sum(be_forward)
            n_other = max(0, len(lc_reqs) + len(active_bes) - 1)
            for be, q_be in zip(active_bes, be_forward):
                m_base = (
                    decode_forward_tokens
                    + lc_forward_tokens
                    + max(0, total_be_forward - q_be)
                )
                be_cost_ms += discount * _bl_eq1_marginal_ms(
                    q_be=q_be,
                    prefix_be=_prefix(be),
                    m_base=m_base,
                    alpha_p=alpha_p,
                    gemm_cost=gemm_cost_fn,
                    n_other=n_other,
                    gamma_launch_ms=float(self._hybrid_gamma_launch_ms),
                    kappa=float(self._hybrid_kappa),
                )
        else:  # flat
            be_cost_ms = discount * flat_marginal * len(active_bes)

        return delta - baseline_ms - be_cost_ms

    def _colo_tbt_slack_gate(
        self,
        candidate: Req,
        candidate_chunk,
        snapshot,
        pending_be_reqs: Sequence[Req] = (),
        in_flight_be_reqs: Optional[Sequence[Req]] = None,
        in_flight_be_cost_state: Optional[Dict[int, tuple]] = None,
        decode_baseline_ms: Optional[float] = None,
        lc_marginal_ms: Optional[float] = None,
    ):
        """Admit candidate iff the tick's TBT slack stays above reserve.

        Colocated ``alg:schedule-hybrid`` counterpart of
        :meth:`_per_lc_slack_gate`.
        Returns ``(admit, reason, predicted_latency_ms, marginal_ms)``.
        """
        estimator = self._smoothagent_admission.estimator
        pred_with = float(
            estimator.estimate(
                decodes=snapshot.decodes,
                prefill_chunks=list(snapshot.lc_chunks)
                + list(snapshot.be_chunks)
                + [candidate_chunk],
            )
        )
        marginal = float(
            os.environ.get("SGLANG_SMOOTHAGENT_BE_MARGINAL_COST_MS", "5.0")
        )
        reserve_ms = self._smoothagent_slack_reserve_ms()
        slack_after = self._colo_tbt_slack_ms(
            extra_be_reqs=list(pending_be_reqs) + [candidate],
            in_flight_be_reqs=in_flight_be_reqs,
            in_flight_be_cost_state=in_flight_be_cost_state,
            decode_baseline_ms=decode_baseline_ms,
            lc_marginal_ms=lc_marginal_ms,
        )
        if slack_after < reserve_ms:
            return (
                False,
                f"tbt_slack<{reserve_ms:.0f} after admit "
                f"(slack={slack_after:.0f}ms)",
                pred_with,
                marginal,
            )
        return True, "tbt slack ok", pred_with, marginal

    # ------------------------------------------------------------------
    # ``alg:schedule-hybrid`` — LC-prefill δ gate (full three-stage admission)
    # ------------------------------------------------------------------

    @staticmethod
    def _colo_tbt_gate_lc_enabled() -> bool:
        """True when the colocated TBT gate is extended to gate LC prefill.

        Opt-in via ``SGLANG_SMOOTHAGENT_COLO_TBT_GATE_LC=1``. It is an
        *extension* of the colocated BE gate, so it requires
        ``SGLANG_SMOOTHAGENT_COLO_TBT_SLACK=1`` as well — without the BE
        gate active the flag is inert (no error). When disabled, every
        LC-prefill gate hook is a no-op and the default scheduler path is
        byte-identical.
        """
        lc = os.environ.get("SGLANG_SMOOTHAGENT_COLO_TBT_GATE_LC", "0") == "1"
        colo = os.environ.get("SGLANG_SMOOTHAGENT_COLO_TBT_SLACK", "0") == "1"
        return lc and colo

    @staticmethod
    def _colo_tbt_lc_gate_passive() -> bool:
        """True when the LC gate runs predictively but never defers.

        Opt-in via ``SGLANG_SMOOTHAGENT_COLO_TBT_LC_GATE_PASSIVE=1``. Only
        meaningful when ``COLO_TBT_GATE_LC=1``. In passive mode the LC
        gate still computes the per-chunk marginal and updates the
        per-tick ``_colo_tbt_lc_marginal_ms`` accumulator (so the BE gate
        keeps seeing route-B baseline = decode + LC marginal), but every
        LC chunk is admitted regardless of predicted overshoot — the
        scheduler effectively only gates BE prefill. Counter
        ``smoothagent_lc_prefill_passive_skips_total`` tracks how often a
        would-be defer was force-admitted.
        """
        return os.environ.get(
            "SGLANG_SMOOTHAGENT_COLO_TBT_LC_GATE_PASSIVE", "0"
        ) == "1"

    @staticmethod
    def _colo_tbt_decode_baseline_ms() -> float:
        """Decode-only tick cost (ms) — the TBT baseline for the LC-prefill gate.

        The cost a tick would have with decode only and no prefill chunk. The
        measured ``predicted_tbt_ms`` EMA cannot serve as this baseline: it
        already absorbs whatever LC prefill the scheduler mixed into recent
        ticks, so adding explicit LC-chunk marginals on top would double-count
        LC prefill.

        It is a **calibrated constant**, not the analytical decode term in
        ``eq:latency-model``.
        On Qwen3-8B / H100 the pure-decode tick latency is essentially flat at
        ~13ms across decode batch size 1-16 and KV 2K-12K — decode is
        memory-bandwidth bound, the per-tick weight load dominates and the
        attention over KV is cheap. The ``alpha_d·Σ kv_len`` term does not
        model this plateau: it scales with Σ kv and mispredicts by MAPE ~155%
        (runs/calibration_decode_baseline_v1/, probe_decode_baseline.py).

        Env override: ``SGLANG_SMOOTHAGENT_COLO_DECODE_BASELINE_MS`` (default
        13.5 — the measured N≥2 mean). Re-calibrate per model/GPU with
        ``scripts/experiments/probe_decode_baseline.py``.
        """
        try:
            return max(
                0.0,
                float(
                    os.environ.get(
                        "SGLANG_SMOOTHAGENT_COLO_DECODE_BASELINE_MS", "13.5"
                    )
                ),
            )
        except (TypeError, ValueError):
            return 13.5

    @staticmethod
    def _colo_tbt_lc_model() -> str:
        """LC-prefill TBT gate model.

        Default ``flat`` preserves the historical path exactly: the LC gate
        adds one independent per-chunk marginal from
        :meth:`_colo_tbt_lc_chunk_marginal_ms` to the decode-only baseline.

        ``hinge_prefix`` is opt-in and uses a calibrated flat-then-ramp
        shape for LC-prefill ticks:

            tick = floor + Σ max(0, chunk_index - hinge) · slope(prefix)
        """
        return os.environ.get(
            "SGLANG_SMOOTHAGENT_COLO_TBT_LC_MODEL", "flat"
        ).lower()

    @staticmethod
    def _colo_tbt_env_float(
        name: str, default: float, *, min_value: float = 0.0
    ) -> float:
        raw = os.environ.get(f"SGLANG_SMOOTHAGENT_{name}", os.environ.get(name))
        try:
            return max(min_value, float(default if raw is None else raw))
        except (TypeError, ValueError):
            return max(min_value, float(default))

    def _colo_tbt_lc_hinge_floor_ms(self) -> float:
        return self._colo_tbt_env_float("COLO_TBT_FLOOR_MS", 14.5)

    def _colo_tbt_lc_hinge_chunks(self) -> int:
        return int(self._colo_tbt_env_float("COLO_TBT_HINGE_CHUNKS", 1.0))

    def _colo_tbt_lc_hinge_chunk_cost_ms(self, req: Req) -> float:
        slope = self._colo_tbt_env_float("COLO_TBT_SLOPE_MS", 4.72)
        prefix_slope = self._colo_tbt_env_float("COLO_TBT_PREFIX_SLOPE_MS", 2.073)
        prefix_norm = self._colo_tbt_env_float(
            "COLO_TBT_PREFIX_NORM_TOKENS", 10283.0, min_value=1.0
        )
        prefix = max(0, int(_bl_estimate_prefix(req)))
        return slope + prefix_slope * float(prefix) / prefix_norm

    def _colo_tbt_lc_chunk_marginal_ms(self, req: Req, chunk_cap: int) -> float:
        """Per-tick marginal cost (ms) of one LC prefill chunk.

        A prefill chunk's cost is a function of ``(q, prefix)`` only, so the
        same calibrated marginal model that scores BE chunks in
        :meth:`_colo_tbt_slack_ms` scores LC chunks here — dispatched on
        ``SGLANG_SMOOTHAGENT_MARGINAL_MODE`` (flat / closed_form / estimator).
        The opt-in colocated LC model ``hinge_prefix`` is only used by the
        LC-prefill gate and leaves the BE gate's marginal path untouched.
        """
        q = self._paper_next_prefill_forward_tokens(req, chunk_cap)
        if q <= 0:
            return 0.0
        if self._colo_tbt_lc_model() == "hinge_prefix":
            return self._colo_tbt_lc_hinge_chunk_cost_ms(req)
        prefix = max(0, int(_bl_estimate_prefix(req)))
        discount = float(
            os.environ.get("SGLANG_SMOOTHAGENT_PRECHARGE_DISCOUNT", "1.0")
        )
        mode = os.environ.get("SGLANG_SMOOTHAGENT_MARGINAL_MODE", "flat").lower()
        if mode == "closed_form":
            return discount * _bl_closed_form_marginal_ms(
                q_be=q,
                prefix_be=prefix,
                c0=float(self._marginal_c0_ms),
                c_attn=float(self._marginal_c_attn),
            )
        if mode == "estimator":
            est = getattr(self._smoothagent_admission, "estimator", None)
            default_alpha_p = (
                float(getattr(est.config, "alpha_p", 5e-6))
                if est is not None
                else 5e-6
            )
            alpha_p = (
                float(self._hybrid_alpha_p_0)
                if self._hybrid_alpha_p_0 is not None
                else default_alpha_p
            )
            gemm_cost_fn = est.gemm_cost if est is not None else (lambda _m: 0.0)
            lc_reqs = [
                r for r in (getattr(self, "chunked_reqs", []) or []) if r is not None
            ]
            be_reqs = [
                b
                for b in (getattr(self, "bg_chunked_reqs", []) or [])
                if b is not None and not getattr(b, "smoothagent_be_paused", False)
            ]
            n_other = max(0, len(lc_reqs) + len(be_reqs) - 1)
            decode_forward_tokens = self._paper_decode_forward_tokens()
            lc_forward_tokens = sum(
                self._paper_next_prefill_forward_tokens(r, chunk_cap)
                for r in lc_reqs
            )
            m_base = decode_forward_tokens + max(0, lc_forward_tokens - q)
            return discount * _bl_eq1_marginal_ms(
                q_be=q,
                prefix_be=prefix,
                m_base=m_base,
                alpha_p=alpha_p,
                gemm_cost=gemm_cost_fn,
                n_other=n_other,
                gamma_launch_ms=float(self._hybrid_gamma_launch_ms),
                kappa=float(self._hybrid_kappa),
            )
        # flat (default — backwards-compatible)
        flat_marginal = float(
            os.environ.get("SGLANG_SMOOTHAGENT_BE_MARGINAL_COST_MS", "5.0")
        )
        return discount * flat_marginal

    def _colo_lc_gate_begin_tick(self) -> None:
        """Reset the per-tick LC-prefill gate state.

        Called once at the start of every prefill-batch build. The gate has
        no cross-tick ledger: the running LC-marginal total and the cached
        decode-only baseline are recomputed fresh each tick.
        """
        self._colo_tbt_lc_marginal_ms = 0.0
        self._colo_tbt_lc_admitted_chunks_this_tick = 0
        if self._colo_tbt_gate_lc_enabled():
            if self._colo_tbt_lc_model() == "hinge_prefix":
                self._colo_tbt_decode_baseline_cached_ms = (
                    self._colo_tbt_lc_hinge_floor_ms()
                )
            else:
                self._colo_tbt_decode_baseline_cached_ms = (
                    self._colo_tbt_decode_baseline_ms()
                )
        else:
            self._colo_tbt_decode_baseline_cached_ms = 0.0

    def _colo_lc_gate_admit_chunk(self, req: Req, chunk_cap: int) -> bool:
        """LC-prefill δ admission for one chunk (continuation or new LC).

        Mirrors the LC-prefill loop in ``alg:schedule-hybrid``: a chunk is
        admitted only while the tick's predicted TBT stays within δ. The
        decision is purely a TBT bound — no TTFT term. Callers process LC
        chunks in arrival (FIFO) order and ``break`` on the first denial, so
        the oldest LC always gets first claim on the budget (liveness — no LC
        starves).

        Returns ``True`` ⇒ admit (the running LC-marginal total is updated);
        ``False`` ⇒ defer (caller must ``break``). When the LC gate is
        disabled this always returns ``True`` (no-op).
        """
        if not self._colo_tbt_gate_lc_enabled():
            return True
        delta = self._colo_tbt_slo_ms()
        reserve = self._smoothagent_slack_reserve_ms()
        admitted_chunks = int(
            getattr(self, "_colo_tbt_lc_admitted_chunks_this_tick", 0)
        )
        if self._colo_tbt_lc_model() == "hinge_prefix":
            hinge_chunks = self._colo_tbt_lc_hinge_chunks()
            marginal = (
                0.0
                if admitted_chunks < hinge_chunks
                else self._colo_tbt_lc_chunk_marginal_ms(req, chunk_cap)
            )
        else:
            marginal = self._colo_tbt_lc_chunk_marginal_ms(req, chunk_cap)
        baseline = float(getattr(self, "_colo_tbt_decode_baseline_cached_ms", 0.0))
        running = float(getattr(self, "_colo_tbt_lc_marginal_ms", 0.0))
        predicted = baseline + running + marginal
        if predicted > delta - reserve:
            if self._colo_tbt_lc_gate_passive():
                # Passive mode: track the would-have-deferred event but admit
                # anyway so the LC prefill flow is uninterrupted. The BE gate
                # still sees the updated lc_marginal (route-B baseline).
                self._smoothagent_lc_prefill_passive_skips_total += 1
                logger.debug(
                    "Colo LC gate PASSIVE force-admit: rid=%s predicted=%.2fms "
                    "delta=%.2fms (decode=%.2f lc_running=%.2f chunk=%.2f)",
                    getattr(req, "rid", "?"), predicted, delta,
                    baseline, running, marginal,
                )
            else:
                self._smoothagent_lc_prefill_deferred_total += 1
                logger.debug(
                    "Colo LC gate deferred LC prefill chunk: rid=%s predicted=%.2fms "
                    "delta=%.2fms (decode=%.2f lc_running=%.2f chunk=%.2f)",
                    getattr(req, "rid", "?"), predicted, delta,
                    baseline, running, marginal,
                )
                return False
        self._colo_tbt_lc_marginal_ms = running + marginal
        self._colo_tbt_lc_admitted_chunks_this_tick = admitted_chunks + 1
        self._smoothagent_lc_prefill_admitted_total += 1
        return True

    def _compute_paper_t_budget_ms(self) -> float:
        """Compute ``alg:schedule-prefill`` ``t_budget`` from active LC reqs.

        The paper's "Q_LC" covers every LC whose TTFT is still ticking — that
        is, anything not yet emitted its first decode token. Concretely:

        1. ``self.chunked_reqs`` — FG LCs currently mid-prefill across ticks.
           These are the requests that are MOST at risk of breaking SLO_TTFT
           (already burned much of their budget on prior ticks). ``self.bg_chunked_reqs``
           is BE; excluded.
        2. ``self.waiting_queue`` — LCs that arrived but not yet started.

        ``EstPrefillLatency(r)`` is the *remaining* prefill latency: number of
        tokens still un-prefilled × per-chunk forward latency × ⌈remaining/chunk_cap⌉.
        Returns ``+inf`` only when both lists are empty.
        """
        chunked = [r for r in getattr(self, "chunked_reqs", []) if r is not None]
        queued = list(self.waiting_queue)
        # In-flight first so the slack formula's "cumulative prefill before j"
        # accounts for them — a fresh queued LC must wait for the in-flight
        # ones to finish before getting its first chunk.
        lc_reqs = chunked + queued
        if not lc_reqs:
            return float("inf")
        estimator = self._smoothagent_admission.estimator
        chunk_cap = max(1, int(self._lookahead_config.bg_max_chunk_tokens))

        def _prefill_latency(r: Req) -> float:
            remaining = int(_bl_remaining_prefill(r))
            if remaining <= 0:
                return 0.0
            next_chunk_size = max(1, self._paper_estimate_next_chunk_tokens(r))
            per_chunk = estimator.estimate(
                decodes=(),
                prefill_chunks=[
                    _BLPrefillChunk(
                        chunk_tokens=next_chunk_size,
                        prefix_kv_len=max(0, self._paper_estimate_prefix_kv_len(r)),
                    )
                ],
            )
            n_remaining_chunks = (remaining + chunk_cap - 1) // chunk_cap
            return float(per_chunk) * n_remaining_chunks

        return self._smoothagent_admission.compute_t_budget_ms(
            lc_reqs,
            now_ms=time.time() * 1000.0,
            prefill_latency_ms=_prefill_latency,
        )

    def _build_paper_batch_snapshot(self) -> BatchSnapshot:
        """Snapshot the current batch in the paper estimator's format.

        Delegates to :func:`smoothagent.snapshot_builder.build_paper_batch_snapshot`,
        which consumes the live mixed-chunk state so the ``eq:latency-model``
        estimate
        tracks what actually runs this tick.
        """
        running = getattr(self, "running_batch", None)
        running_reqs = list(getattr(running, "reqs", []) or [])
        try:
            tbt_ms = float(self._latency_predictor.predicted_tbt_ms or 0.0)
        except AttributeError:
            tbt_ms = 0.0
        return _bl_build_snapshot(
            running_reqs=running_reqs,
            waiting_queue=list(self.waiting_queue),
            bg_chunked_reqs=list(getattr(self, "bg_chunked_reqs", []) or []),
            bg_waiting_queue=list(getattr(self, "bg_waiting_queue", []) or []),
            bg_max_chunk_tokens=int(self._lookahead_config.bg_max_chunk_tokens),
            predicted_tbt_ms=tbt_ms,
            chunk_mixer_compute=getattr(
                self._chunk_mixer, "compute_chunk_size", None
            ),
        )

    def predict_batch_latency_ms(self) -> float:
        """``eq:latency-model`` prediction for the next tick's batch latency.

        Uses the current scheduler state via :meth:`_build_paper_batch_snapshot`.
        Returns ``0.0`` when the SmoothAgent admission controller is not
        initialized (e.g., BG scheduling disabled).
        """
        ctrl = getattr(self, "_smoothagent_admission", None)
        if ctrl is None:
            return 0.0
        snapshot = self._build_paper_batch_snapshot()
        return ctrl.estimator.estimate(
            decodes=snapshot.decodes,
            prefill_chunks=list(snapshot.lc_chunks) + list(snapshot.be_chunks),
        )

    # ── Request classification ─────────────────────────────────────

    def classify_request(self, req: Req) -> None:
        """Classify request as FG or BG.

        Honors the SmoothAgent ``smoothagent_priority_class``
        field (``"be"`` ⇒ BACKGROUND) when set, and falls back to the
        legacy ``request_class`` hint (``"bg"`` ⇒ BACKGROUND) otherwise.
        ``is_lookahead=True`` is treated equivalently to ``priority_class="be"``.
        """
        priority_class = getattr(req, "smoothagent_priority_class", None)
        is_lookahead = getattr(req, "smoothagent_is_lookahead", None)
        if priority_class == "be" or is_lookahead is True:
            req.request_channel = RequestChannel.BACKGROUND
            return
        if priority_class == "lc":
            req.request_channel = RequestChannel.FOREGROUND
            return
        request_class = getattr(req, '_request_class_hint', None)
        if request_class == "bg":
            req.request_channel = RequestChannel.BACKGROUND
        else:
            req.request_channel = RequestChannel.FOREGROUND

    def _get_lowest_lookahead_priority(self) -> int:
        """Return the least-preferred priority under the active scheduler ordering."""
        if getattr(self, "schedule_low_priority_values_first", False):
            return sys.maxsize
        return -sys.maxsize - 1

    def _get_highest_lookahead_priority(self) -> int:
        """Return the most-preferred priority under the active scheduler ordering."""
        if getattr(self, "schedule_low_priority_values_first", False):
            return -sys.maxsize - 1
        return sys.maxsize

    def _clamp_bg_chunk_under_fg_pressure(self, c_chunk: int, fg_queue_len: int) -> int:
        """Keep BG prefill tiny whenever there is any pending foreground backlog."""
        if fg_queue_len <= 0:
            return c_chunk

        align_size = getattr(self, "truncation_align_size", 1)
        align = max(1, align_size or 1)
        pressure_cap = max(align, self._lookahead_config.bg_max_chunk_tokens // 16)
        return min(c_chunk, pressure_cap)

    def _has_bg_req_slot_capacity(self, adder) -> bool:
        """Return whether this tick can add another BG request slot."""
        get_num_allocatable_reqs = getattr(self, "get_num_allocatable_reqs", None)
        if not callable(get_num_allocatable_reqs):
            return True
        running_batch = getattr(self, "running_batch", None)
        running_bs = len(getattr(running_batch, "reqs", []) or [])
        if len(adder.can_run_list) >= get_num_allocatable_reqs(running_bs):
            return False
        req_pool = getattr(self, "req_to_token_pool", None)
        if req_pool is not None:
            return len(adder.can_run_list) < req_pool.available_size()
        return True

    def _bg_inflight_req_limit(self) -> int:
        limit = int(getattr(self._lookahead_config, "bg_max_inflight_reqs", 2) or 2)
        return max(1, limit)

    def _has_bg_inflight_capacity(self) -> bool:
        self.clear_terminal_promoted_lookahead_req()
        inflight = len(getattr(self, "bg_chunked_reqs", []) or [])
        if getattr(self, "promoted_lookahead_req", None) is not None:
            inflight += 1
        inflight += sum(
            1
            for req in getattr(self, "waiting_queue", []) or []
            if getattr(req, "lookahead_promoted", False)
        )
        return inflight < self._bg_inflight_req_limit()

    def prune_released_bg_chunked_reqs(self) -> None:
        self.bg_chunked_reqs = [
            req
            for req in getattr(self, "bg_chunked_reqs", []) or []
            if getattr(req, "req_pool_idx", None) is not None
            or getattr(req, "lookahead_pending_more_chunks", False)
        ]

    def clear_terminal_promoted_lookahead_req(self) -> None:
        req = getattr(self, "promoted_lookahead_req", None)
        if req is None:
            pass
        else:
            if getattr(req, "finished_reason", None) is not None and not getattr(
                req, "lookahead_complete_handled", False
            ):
                self.on_bg_request_complete(req)
            if getattr(req, "lookahead_complete_handled", False):
                self.promoted_lookahead_req = None

        waiting_queue = getattr(self, "waiting_queue", None)
        if not waiting_queue:
            return
        kept = []
        changed = False
        for queued_req in waiting_queue:
            if not getattr(queued_req, "lookahead_promoted", False):
                kept.append(queued_req)
                continue
            if getattr(queued_req, "finished_reason", None) is not None and not getattr(
                queued_req, "lookahead_complete_handled", False
            ):
                self.on_bg_request_complete(queued_req)
            if getattr(queued_req, "lookahead_complete_handled", False):
                changed = True
                continue
            kept.append(queued_req)
        if changed:
            self.waiting_queue = kept

    def preempt_bg_lookahead_for_foreground(self) -> None:
        """Drop speculative BG chunks when foreground requests need req slots."""
        fg_waiting = sum(
            1
            for req in getattr(self, "waiting_queue", []) or []
            if req.reduce_request_type != ReduceRequestType.LOOKAHEAD_REDUCE
        )
        if fg_waiting <= 0 or not self.bg_chunked_reqs:
            return

        available = self.req_to_token_pool.available_size()
        needed_now = min(fg_waiting, self._bg_inflight_req_limit())
        if available >= needed_now:
            return

        dropped = 0
        for req in list(self.bg_chunked_reqs):
            session = self.sessions.get(req.session_id)
            if session is not None:
                session.clear_lookahead(req.lookahead_task_id or None, self.tree_cache)
            self._release_dropped_lookahead_req(req)
            dropped += 1
        self.bg_chunked_reqs = []

        if dropped:
            logger.info(
                "Preempted %d BG lookahead chunk(s) for foreground backlog: "
                "fg_waiting=%d req_slots_available=%d",
                dropped,
                fg_waiting,
                available,
            )

    def _is_target_lookahead_req(
        self,
        req: Optional[Req],
        session_id: str,
        task_id: str,
        version: int,
        source_ctx_len_tokens: Optional[int] = None,
    ) -> bool:
        return (
            req is not None
            and req.reduce_request_type == ReduceRequestType.LOOKAHEAD_REDUCE
            and req.session_id == session_id
            and req.lookahead_task_id == task_id
            and (
                source_ctx_len_tokens is None
                or req.lookahead_source_ctx_len_tokens == source_ctx_len_tokens
            )
        )

    def _is_stale_lookahead_req(
        self,
        req: Req,
        session,
    ) -> bool:
        # External BE requests from API-based serving don't have a
        # control-plane lookahead session: they arrive directly via
        # /generate with `is_lookahead=true` and carry no
        # ``lookahead_task_id``. They are NOT stale — there is no version /
        # ctx-len drift to check. Only control-plane lookahead reqs (those
        # with a non-empty ``lookahead_task_id``) need the staleness check.
        if not getattr(req, "lookahead_task_id", ""):
            return False
        if session is None:
            return True
        branch = session.get_lookahead_branch(task_id=req.lookahead_task_id)
        active_source_ctx_len = None
        if branch is not None:
            active_source_ctx_len = branch.source_ctx_len_tokens
        elif session.lookahead_pending_task_id == req.lookahead_task_id:
            active_source_ctx_len = session.lookahead_pending_ctx_len

        stale_version = req.lookahead_version != session.conversation_version and branch is None
        stale_ctx_len = (
            active_source_ctx_len is not None
            and active_source_ctx_len != 0
            and req.lookahead_source_ctx_len_tokens != active_source_ctx_len
        )
        active_task_id = None
        if branch is not None:
            active_task_id = branch.task_id
        else:
            active_task_id = session.get_pending_lookahead_task()
            if active_task_id is None:
                active_task_id = session.lookahead_commit_task_id
            if (
                active_task_id is None
                and session.lookahead_result is not None
                and session.lookahead_result.version == session.conversation_version
            ):
                active_task_id = session.lookahead_result.task_id
        stale_task = active_task_id is not None and active_task_id != req.lookahead_task_id
        return stale_version or stale_ctx_len or stale_task

    def _mark_promoted_lookahead_req(self, req: Req) -> None:
        req.lookahead_promoted = True
        req.request_channel = RequestChannel.BACKGROUND
        req.priority = self._get_highest_lookahead_priority()

    def _enqueue_promoted_waiting_req(self, req: Req) -> None:
        self._mark_promoted_lookahead_req(req)
        if req not in self.waiting_queue:
            self.waiting_queue.append(req)

    def _find_inflight_lookahead_req(
        self,
        session_id: str,
        task_id: str,
        version: int,
        source_ctx_len_tokens: Optional[int] = None,
    ) -> Optional[Req]:
        batches = []
        if getattr(self, "cur_batch", None) is not None:
            batches.append(self.cur_batch)
        if getattr(self, "last_batch", None) is not None:
            batches.append(self.last_batch)
        if hasattr(self, "result_queue"):
            batches.extend(batch for batch, _ in self.result_queue)

        for batch in batches:
            for req in batch.reqs:
                if req.finished():
                    continue
                if self._is_target_lookahead_req(
                    req,
                    session_id,
                    task_id,
                    version,
                    source_ctx_len_tokens,
                ):
                    return req
        return None

    def _find_mutable_local_lookahead_req(
        self,
        session_id: str,
        task_id: str,
    ) -> Optional[Req]:
        candidates: List[Req] = []
        if self.promoted_lookahead_req is not None:
            candidates.append(self.promoted_lookahead_req)
        # Iterate every in-flight BG chunked req (paper-mode multi).
        candidates.extend(self.bg_chunked_reqs)
        candidates.extend(self.bg_waiting_queue)
        candidates.extend(self.waiting_queue)

        for req in candidates:
            if (
                req.session_id == session_id
                and req.reduce_request_type == ReduceRequestType.LOOKAHEAD_REDUCE
                and req.lookahead_task_id == task_id
            ):
                return req
        return None

    def _sync_lookahead_req_to_branch(
        self,
        req: Req,
        branch,
    ) -> bool:
        target_input_ids = list(branch.input_ids)
        current_input_ids = list(req.origin_input_ids)
        if (
            len(target_input_ids) < len(current_input_ids)
            or current_input_ids != target_input_ids[: len(current_input_ids)]
        ):
            return False

        req.origin_input_ids = target_input_ids
        req.origin_input_ids_unpadded = tuple(target_input_ids)
        req.origin_input_text = branch.input_text
        req.lookahead_source_ctx_len_tokens = branch.source_ctx_len_tokens
        req.lookahead_artifact = dict(branch.artifact)
        req.ctx_len_tokens = branch.source_ctx_len_tokens

        req.fill_ids = req.origin_input_ids + req.output_ids
        prefix_len = len(req.prefix_indices) if req.prefix_indices is not None else 0
        req.set_extend_input_len(max(0, len(req.fill_ids) - prefix_len))
        return True

    def _sync_pending_lookahead_req_from_branch(
        self,
        session,
        req: Req,
    ) -> bool:
        branch = session.get_lookahead_branch(task_id=req.lookahead_task_id)
        if branch is None:
            return False
        if branch.source_ctx_len_tokens <= req.lookahead_source_ctx_len_tokens:
            return True
        return self._sync_lookahead_req_to_branch(req, branch)

    def promote_lookahead_for_commit(
        self,
        session,
        task_id: str,
    ) -> str:
        """
        Atomically move a pending lookahead reduce job onto the hard-commit fast lane.

        Returns:
          "promoted": moved from BG queue/chunk lane to promoted slot.
          "running": already in-flight on GPU; commit is latched and will complete on ready.
          "missing": no pending or in-flight job matches the requested task.
        """
        self.clear_terminal_promoted_lookahead_req()
        session_id = session.session_id
        version = session.conversation_version
        source_ctx_len_tokens = session.lookahead_pending_ctx_len or (
            session.get_lookahead_branch(task_id=task_id).source_ctx_len_tokens
            if session.get_lookahead_branch(task_id=task_id) is not None
            else None
        )
        active_promoted_req = self.promoted_lookahead_req

        if self._is_target_lookahead_req(
            active_promoted_req,
            session_id,
            task_id,
            version,
            source_ctx_len_tokens,
        ):
            self._mark_promoted_lookahead_req(active_promoted_req)
            return "promoted"

        # Search every in-flight BG chunked req (paper-mode multi).
        for i, r in enumerate(list(self.bg_chunked_reqs)):
            if self._is_target_lookahead_req(
                r,
                session_id,
                task_id,
                version,
                source_ctx_len_tokens,
            ):
                req = self.bg_chunked_reqs.pop(i)
                if active_promoted_req is not None:
                    self._enqueue_promoted_waiting_req(req)
                    return "queued"
                self.promoted_lookahead_req = req
                self._mark_promoted_lookahead_req(req)
                return "promoted"

        for i, req in enumerate(self.bg_waiting_queue):
            if self._is_target_lookahead_req(
                req,
                session_id,
                task_id,
                version,
                source_ctx_len_tokens,
            ):
                req = self.bg_waiting_queue.pop(i)
                self._anti_starvation.on_dequeue(req.rid)
                if active_promoted_req is not None:
                    self._enqueue_promoted_waiting_req(req)
                    return "queued"
                self.promoted_lookahead_req = req
                self._mark_promoted_lookahead_req(req)
                return "promoted"

        for i, req in enumerate(self.waiting_queue):
            if self._is_target_lookahead_req(
                req,
                session_id,
                task_id,
                version,
                source_ctx_len_tokens,
            ):
                if active_promoted_req is not None:
                    self._mark_promoted_lookahead_req(req)
                    return "queued"
                self.promoted_lookahead_req = self.waiting_queue.pop(i)
                self._mark_promoted_lookahead_req(req)
                return "promoted"

        inflight_req = self._find_inflight_lookahead_req(
            session_id,
            task_id,
            version,
            source_ctx_len_tokens,
        )
        if inflight_req is not None:
            self._mark_promoted_lookahead_req(inflight_req)
            if getattr(inflight_req, "lookahead_pending_more_chunks", False):
                if self.promoted_lookahead_req is None:
                    self.promoted_lookahead_req = inflight_req
                    return "promoted"
                if self.promoted_lookahead_req is not inflight_req:
                    self._enqueue_promoted_waiting_req(inflight_req)
                    return "queued"
            return "running"

        return "missing"

    def _drop_local_lookahead_task(
        self,
        session_id: str,
        task_id: Optional[str] = None,
    ) -> None:
        def matches(req: Optional[Req]) -> bool:
            if req is None or req.session_id != session_id:
                return False
            if task_id is None:
                return req.reduce_request_type == ReduceRequestType.LOOKAHEAD_REDUCE
            return (
                req.reduce_request_type == ReduceRequestType.LOOKAHEAD_REDUCE
                and req.lookahead_task_id == task_id
            )

        def drop_if_match(req: Optional[Req]) -> bool:
            if not matches(req):
                return False
            self._release_dropped_lookahead_req(req)
            return True

        self.bg_waiting_queue = [
            req for req in self.bg_waiting_queue if not drop_if_match(req)
        ]
        self.waiting_queue = [
            req for req in self.waiting_queue if not drop_if_match(req)
        ]

        # Drop every matching BG chunked req (paper-mode multi).
        self.bg_chunked_reqs = [
            r for r in self.bg_chunked_reqs if not drop_if_match(r)
        ]
        if matches(self.promoted_lookahead_req):
            self._release_dropped_lookahead_req(self.promoted_lookahead_req)
            self.promoted_lookahead_req = None

    def _release_dropped_lookahead_req(self, req: Optional[Req]) -> None:
        """Free local resources for a discarded lookahead BG request."""
        if req is None:
            return
        req.lookahead_pending_more_chunks = False
        sender = getattr(req, "disagg_kv_sender", None)
        if sender is not None and hasattr(sender, "abort"):
            try:
                sender.abort()
            except Exception:
                logger.debug("Failed to abort dropped lookahead KV sender", exc_info=True)
        if getattr(req, "req_pool_idx", None) is None:
            return
        try:
            release_kv_cache(req, self.tree_cache, is_insert=False)
        except Exception:
            logger.warning(
                "Failed to release dropped lookahead req: rid=%s",
                getattr(req, "rid", None),
                exc_info=True,
            )

    # ── Phase-Offline: schedule BG in each tick ──────────────────────

    def invalidate_stale_lookaheads(self) -> None:
        """Drop background work that no longer matches the live session version."""
        if not hasattr(self, "bg_waiting_queue") or not hasattr(
            self, "bg_chunked_reqs"
        ):
            return

        active_queue: List[Req] = []
        for req in self.bg_waiting_queue:
            session = self.sessions.get(req.session_id) if req.session_id is not None else None
            if self._is_stale_lookahead_req(req, session):
                if session is not None:
                    session.clear_lookahead(req.lookahead_task_id or None, self.tree_cache)
                self._release_dropped_lookahead_req(req)
                continue
            active_queue.append(req)
        self.bg_waiting_queue = active_queue

        active_waiting_queue: List[Req] = []
        for req in self.waiting_queue:
            if req.reduce_request_type != ReduceRequestType.LOOKAHEAD_REDUCE:
                active_waiting_queue.append(req)
                continue
            session = self.sessions.get(req.session_id) if req.session_id is not None else None
            if self._is_stale_lookahead_req(req, session):
                if session is not None:
                    session.clear_lookahead(req.lookahead_task_id or None, self.tree_cache)
                self._release_dropped_lookahead_req(req)
                continue
            active_waiting_queue.append(req)
        self.waiting_queue = active_waiting_queue

        # Drop stale BG chunked reqs (paper-mode multi).
        active_bg_chunked: List[Req] = []
        for r in self.bg_chunked_reqs:
            session = self.sessions.get(r.session_id)
            if self._is_stale_lookahead_req(r, session):
                if session is not None:
                    session.clear_lookahead(
                        r.lookahead_task_id or None,
                        self.tree_cache,
                    )
                self._release_dropped_lookahead_req(r)
                continue
            active_bg_chunked.append(r)
        self.bg_chunked_reqs = active_bg_chunked

        if self.promoted_lookahead_req is not None:
            session = self.sessions.get(self.promoted_lookahead_req.session_id)
            if self._is_stale_lookahead_req(self.promoted_lookahead_req, session):
                if session is not None:
                    session.clear_lookahead(
                        self.promoted_lookahead_req.lookahead_task_id or None,
                        self.tree_cache,
                    )
                self._release_dropped_lookahead_req(self.promoted_lookahead_req)
                self.promoted_lookahead_req = None

    def schedule_lookahead_tick(self) -> List[Req]:
        """
        Called from get_next_batch_to_run to get BG reqs to route to waiting_queue.
        Returns list of BG reqs to schedule this tick; empty when no BG or not enabled.
        """
        if self.bg_chunked_req is not None or not self.bg_waiting_queue:
            return []
        req = self.bg_waiting_queue[0]
        if not self.gate_bg_request(req):
            return []
        self.bg_waiting_queue.pop(0)
        self._anti_starvation.on_dequeue(req.rid)
        return [req]

    def schedule_bg_scheduling_tick(self):
        """
        Called at the end of each scheduling cycle (after FG batch is built).
        Implements the two-phase scheduling: Phase-Online (FG) is already done
        by the existing scheduler; this method implements Phase-Offline (BG).

        Handles: BG prefill chunked append + BG decode gate.
        """
        if not hasattr(self, '_slo_budget_ctrl'):
            return

        # Tick the burst guard cooldown
        self._burst_guard.tick()

        # Update stats
        self._bg_stats.update_queue_state(
            len(self.bg_waiting_queue),
            self.bg_chunked_req is not None,
        )

    def maybe_append_bg_prefill(self, adder):
        """
        Phase-Offline prefill (``alg:schedule-hybrid`` BE-prefill loop):
        after Phase-Online (FG scheduling) selects the foreground prefill
        list, admit as many BE chunks as the per-tick batch budget allows
        — each gated by ``eq:latency-model`` (predicted batch latency ≤ TBT
        bound).

        Two modes:

        * **paper-mode** (``SGLANG_SMOOTHAGENT_PAPER_MODE=1``) — multi-admit
          while-loop. Continues every in-flight ``bg_chunked_reqs`` first
          (mirror of FG continuation), then iterates the BG waiting queue
          and admits chunks while: (a) FG-backlog and KV-pressure guards
          still pass, (b) the latency-model gate admits the candidate,
          (c) the adder's
          shared budget hasn't been exhausted.
        * **legacy** (paper-mode off) — the SLOBudgetController +
          gate_bg_request single-admit flow is preserved unchanged for
          deployments that haven't migrated to paper-mode.
        """
        if not hasattr(self, '_slo_budget_ctrl'):
            return []

        # Burst guard applies in both modes.
        if self._burst_guard.in_cooldown:
            return []

        paper_mode = getattr(self, '_smoothagent_paper_mode', False)

        if paper_mode:
            return self._maybe_append_bg_prefill_paper(adder)
        return self._maybe_append_bg_prefill_legacy(adder)

    def _should_wait_for_bg_queue_accumulation(self) -> bool:
        """Experiment-only queue barrier for controlled latency probes."""
        try:
            max_wait_ms = float(
                os.environ.get(
                    "SGLANG_SMOOTHAGENT_BG_QUEUE_ACCUM_MAX_WAIT_MS", "0"
                )
                or 0.0
            )
        except ValueError:
            max_wait_ms = 0.0
        try:
            min_queue = int(
                os.environ.get("SGLANG_SMOOTHAGENT_BG_QUEUE_ACCUM_MIN", "0") or 0
            )
        except ValueError:
            min_queue = 0
        if max_wait_ms <= 0.0 or min_queue <= 1:
            return False

        queue_len = len(getattr(self, "bg_waiting_queue", []) or [])
        if queue_len <= 0:
            self._bg_waiting_queue_first_enqueue_perf = None
            return False
        if queue_len >= min_queue:
            return False

        now = time.perf_counter()
        first_enqueue = getattr(self, "_bg_waiting_queue_first_enqueue_perf", None)
        if first_enqueue is None:
            first_enqueue = now
            self._bg_waiting_queue_first_enqueue_perf = first_enqueue
        return (now - float(first_enqueue)) * 1000.0 < max_wait_ms

    def _maybe_append_bg_prefill_paper(self, adder):
        """
        ``alg:schedule-hybrid`` BE-prefill multi-admit loop. Continues
        every in-flight ``bg_chunked_reqs`` (multi-chunked across ticks),
        then admits as many fresh BE chunks as guards + latency-model gate allow.

        When ``SGLANG_SMOOTHAGENT_DISAGG_SIM=1`` is set, the admission gate
        switches to ``alg:schedule-prefill`` (TTFT budget) instead of
        ``alg:schedule-hybrid`` (TBT).
        Used for single-GPU PD-prefill simulation when no real disaggregated
        cluster is available.
        """
        # Gate mode toggles:
        #   - PER_LC_BUDGET=1 → global per-LC time ledger (strictest, ours)
        #   - DISAGG_SIM=1   → ``alg:schedule-prefill`` (TTFT t_budget)
        #   - default        → ``alg:schedule-hybrid`` (TBT delta)
        per_lc_budget = os.environ.get("SGLANG_SMOOTHAGENT_PER_LC_BUDGET", "0") == "1"
        disagg_sim = os.environ.get("SGLANG_SMOOTHAGENT_DISAGG_SIM", "0") == "1"
        #   - COLO_TBT_SLACK=1 → colocated ``alg:schedule-hybrid`` gate
        colo_tbt = os.environ.get("SGLANG_SMOOTHAGENT_COLO_TBT_SLACK", "0") == "1"
        t_budget_ms_const: Optional[float] = None
        in_flight_lcs: List[Req] = []
        # When the colocated LC-prefill gate is active, the BE slack must be
        # taken against the decode-only baseline + this tick's LC-prefill
        # marginal (decided earlier in the prefill build). ``None`` ⇒ measured
        # EMA baseline (LC gate off — BE-only behavior unchanged).
        colo_decode_baseline: Optional[float] = None
        colo_lc_marginal: Optional[float] = None
        if per_lc_budget:
            # Gather in-flight LCs (chunked) + queued LCs that will run soon.
            # No persistent ledger init — slack is recomputed each tick.
            for r in getattr(self, "chunked_reqs", []):
                if r is None:
                    continue
                in_flight_lcs.append(r)
            for r in self.waiting_queue:
                in_flight_lcs.append(r)

            # Reset paused flag — preemption is decided fresh every tick.
            for be in getattr(self, "bg_chunked_reqs", []) or []:
                if be is not None:
                    be.smoothagent_be_paused = False

            # Surface min slack as the t_budget stat for monitoring.
            if in_flight_lcs:
                min_slack = min(self._per_lc_slack_ms(lc) for lc in in_flight_lcs)
                self._smoothagent_last_t_budget_ms = float(min_slack)

            # ── Preemption: pause longest BEs while any LC slack < 0 ──
            # Iterate from longest BE to shortest, marking paused. Recompute
            # min slack after each pause (paused BE no longer contributes to
            # cost). Stop once all LCs are non-negative. In strict per-LC
            # budget mode, preserving a BE for queue drainage after slack is
            # already negative systematically spends LC TTFT budget.
            preempt_enabled = os.environ.get(
                "SGLANG_SMOOTHAGENT_BE_PREEMPT", "0"
            ) == "1"
            if preempt_enabled and in_flight_lcs:
                reserve_ms = self._smoothagent_slack_reserve_ms()
                bes = [b for b in (self.bg_chunked_reqs or []) if b is not None]
                bes_longest_first = sorted(
                    bes,
                    key=lambda r: -int(_bl_remaining_prefill(r)),
                )
                for be in bes_longest_first:
                    if all(
                        self._per_lc_slack_ms(lc) >= reserve_ms
                        for lc in in_flight_lcs
                    ):
                        break
                    be.smoothagent_be_paused = True
        elif disagg_sim:
            t_budget_ms_const = self._compute_paper_t_budget_ms()
            if math.isfinite(t_budget_ms_const):
                self._smoothagent_last_t_budget_ms = float(t_budget_ms_const)
        elif colo_tbt:
            # Colocated ``alg:schedule-hybrid`` TBT slack gate. Reset paused flags,
            # surface the tick's TBT slack as the t_budget stat, then
            # preempt longest BEs while the tick is over the TBT budget.
            # Mirrors the per-LC budget path but with a per-tick TBT
            # bound instead of cumulative TTFT.
            #
            # If the LC-prefill gate is also on, the LC gate already ran in
            # the prefill build (scheduler.py LC loops, before this call);
            # ``_colo_tbt_lc_marginal_ms`` holds this tick's admitted LC
            # marginal. Feed that + the decode-only baseline to the BE slack
            # so LC prefill is counted once, not double-counted via the EMA.
            if self._colo_tbt_gate_lc_enabled():
                colo_decode_baseline = float(
                    getattr(self, "_colo_tbt_decode_baseline_cached_ms", 0.0)
                )
                colo_lc_marginal = float(
                    getattr(self, "_colo_tbt_lc_marginal_ms", 0.0)
                )
            for be in getattr(self, "bg_chunked_reqs", []) or []:
                if be is not None:
                    be.smoothagent_be_paused = False
            self._smoothagent_last_t_budget_ms = float(
                self._colo_tbt_slack_ms(
                    decode_baseline_ms=colo_decode_baseline,
                    lc_marginal_ms=colo_lc_marginal,
                )
            )
            preempt_enabled = os.environ.get(
                "SGLANG_SMOOTHAGENT_BE_PREEMPT", "0"
            ) == "1"
            if preempt_enabled:
                reserve_ms = self._smoothagent_slack_reserve_ms()
                bes_longest_first = sorted(
                    [b for b in (self.bg_chunked_reqs or []) if b is not None],
                    key=lambda r: -int(_bl_remaining_prefill(r)),
                )
                for be in bes_longest_first:
                    if self._colo_tbt_slack_ms(
                        decode_baseline_ms=colo_decode_baseline,
                        lc_marginal_ms=colo_lc_marginal,
                    ) >= reserve_ms:
                        break
                    be.smoothagent_be_paused = True

        c_chunk = int(self._lookahead_config.bg_max_chunk_tokens)
        # ChunkMixer TBT guidance.
        if self._latency_predictor.tbt_sample_count > 0:
            c_chunk = self._chunk_mixer.compute_chunk_size(
                self._latency_predictor.predicted_tbt_ms,
                base_chunk=c_chunk,
            )
        c_chunk = self._clamp_bg_chunk_under_fg_pressure(
            c_chunk, len(self.waiting_queue)
        )
        if c_chunk <= 0:
            return []
        scheduled_bg_chunked_reqs: List[Req] = []

        truncation_align_size = getattr(self, 'truncation_align_size', 1)
        fg_backlog_len = len(self.waiting_queue)
        pause_bg_on_fg_queue = (
            os.environ.get("SGLANG_SMOOTHAGENT_PAUSE_BG_ON_FG_QUEUE", "0") == "1"
            and fg_backlog_len > 0
        )

        # ── (1) Continue every in-flight BG chunked req this tick.
        # Per-tick slack model has no ledger: continuing BEs are already
        # accounted for via ``bg_chunked_reqs`` in the slack formula.
        bg_cost_reqs_before_continuation: List[Req] = []
        bg_cost_state_before_continuation: Dict[int, tuple] = {}
        for r in self.bg_chunked_reqs:
            if (
                r is None
                or pause_bg_on_fg_queue
                or getattr(r, "smoothagent_be_paused", False)
            ):
                continue
            remaining = max(0, int(_bl_remaining_prefill(r)))
            if remaining <= 0:
                continue
            bg_cost_reqs_before_continuation.append(r)
            bg_cost_state_before_continuation[id(r)] = (
                remaining,
                max(0, int(_bl_estimate_prefix(r))),
            )
        new_bg_chunked: List[Req] = []
        for r in self.bg_chunked_reqs:
            if not self._has_bg_req_slot_capacity(adder):
                new_bg_chunked.append(r)
                continue
            # Preemption: skip continuation for paused BE chunks. KV stays
            # allocated (req remains in bg_chunked_reqs), is_chunked counter
            # is NOT incremented (no forward pass), so output processor
            # invariant is preserved. Pause flag is reset at the top of next
            # tick and re-decided fresh.
            if getattr(r, "smoothagent_be_paused", False):
                new_bg_chunked.append(r)
                continue
            if pause_bg_on_fg_queue:
                new_bg_chunked.append(r)
                continue
            session = self.sessions.get(r.session_id)
            if session is not None:
                self._sync_pending_lookahead_req_from_branch(session, r)
            r.init_next_round_input(self.tree_cache)
            prev_can_run_len = len(adder.can_run_list)
            kept = adder.add_chunked_req(r, chunk_cap=c_chunk)
            if len(adder.can_run_list) > prev_can_run_len:
                tokens_added = r.extend_input_len
                self._bg_stats.bg_prefill_tokens_total += tokens_added
                self._bg_stats.bg_prefill_tokens_last_tick = tokens_added
                r.is_chunked += 1
            if kept is not None:
                r.lookahead_pending_more_chunks = True
                new_bg_chunked.append(r)
                if len(adder.can_run_list) > prev_can_run_len:
                    scheduled_bg_chunked_reqs.append(r)
            elif len(adder.can_run_list) > prev_can_run_len:
                r.lookahead_pending_more_chunks = False
        self.bg_chunked_reqs = new_bg_chunked

        if self._should_wait_for_bg_queue_accumulation():
            return scheduled_bg_chunked_reqs

        # ── (2) Admit fresh BE candidates while the latency model + guards pass.
        #
        # Each iteration:
        #   - Re-check FG-backlog + KV-pressure guards (state changes as
        #     LC progresses across ticks).
        #   - Pop next BG candidate from the queue.
        #   - Build a fresh batch snapshot reflecting all admits so far.
        #   - Run ``eq:latency-model`` chunk-level admission against the snapshot.
        #   - If admitted, call adder.add_one_req(chunk_cap=c_chunk).
        #
        # The shared adder budget naturally bounds the loop: when
        # ``rem_chunk_tokens`` drops below the BE chunk's truncation
        # alignment, add_one_req returns OTHER and we break.
        # ``pending_be_this_tick``: BEs admitted in earlier iterations of
        # this loop. The slack gate counts them as if already running so
        # the second admission sees the cost of the first.
        pending_be_this_tick: List[Req] = []
        while True:
            if not self._has_bg_inflight_capacity():
                break
            if not self._has_bg_req_slot_capacity(adder):
                break
            free_blocks = self.token_to_kv_pool_allocator.available_size()
            total_blocks = self.token_to_kv_pool_allocator.size
            fg_queue_len = len(self.waiting_queue)

            if fg_queue_len > self._lookahead_config.bg_admission_fg_backlog_threshold:
                break
            if total_blocks > 0:
                # Subtract radix-cache evictable pages from "used": those can
                # be reclaimed on demand and aren't real KV pressure. Without
                # this subtraction the guard trips at ~0% real usage once
                # the radix cache fills up (it never voluntarily evicts).
                evictable = 0
                try:
                    evictable = int(self.tree_cache.evictable_size())
                except Exception:
                    pass
                allocated_blocks = max(0, total_blocks - free_blocks)
                # The radix cache may temporarily over-report evictable logical
                # tokens after chunked lookahead/session reuse. Admission only
                # needs physical pressure, so never subtract more evictable
                # tokens than are actually allocated from the KV pool.
                evictable = min(evictable, allocated_blocks)
                effective_used = allocated_blocks - evictable
                kv_used_ratio = effective_used / total_blocks
                if kv_used_ratio > self._lookahead_config.bg_admission_kv_pressure_ratio:
                    break
            if not self.bg_waiting_queue:
                break

            candidate = self.bg_waiting_queue[0]
            session = self.sessions.get(candidate.session_id)
            if session is not None:
                self._sync_pending_lookahead_req_from_branch(session, candidate)
            candidate.init_next_round_input(self.tree_cache)

            # ``eq:latency-model`` chunk-level admission gate, against fresh snapshot.
            ext = int(getattr(candidate, 'extend_input_len', 0) or 0)
            chunk_tokens = min(c_chunk, ext) if ext > 0 else c_chunk
            candidate_chunk = _BLPrefillChunk(
                chunk_tokens=max(1, chunk_tokens),
                prefix_kv_len=max(0, _bl_estimate_prefix(candidate)),
            )
            snapshot = self._build_paper_batch_snapshot()
            if per_lc_budget and in_flight_lcs:
                ok, reason, pred_with, _marginal = self._per_lc_slack_gate(
                    candidate,
                    candidate_chunk,
                    snapshot,
                    in_flight_lcs,
                    pending_be_reqs=pending_be_this_tick,
                    in_flight_be_reqs=bg_cost_reqs_before_continuation,
                    in_flight_be_cost_state=bg_cost_state_before_continuation,
                )
                # Synthesize a verdict so existing logging/counters work.
                from sglang.srt.managers.smoothagent.admission_controller import (
                    AdmissionVerdict as _Verdict,
                )
                verdict = _Verdict(
                    admit=ok,
                    reason=reason,
                    predicted_latency_ms=pred_with,
                    t_budget_ms=None,
                )
            elif per_lc_budget and not in_flight_lcs:
                # No LC to protect → admit BE candidate unconditionally. The
                # per-LC slack gate has nothing to compute; the disaggregated
                # ``disaggregated`` path needs a t_budget that we never
                # initialized in this combined mode, so synthesize an admit.
                from sglang.srt.managers.smoothagent.admission_controller import (
                    AdmissionVerdict as _Verdict,
                )
                verdict = _Verdict(
                    admit=True,
                    reason="no in-flight LC; per-LC budget vacuously satisfied",
                    predicted_latency_ms=0.0,
                    t_budget_ms=None,
                )
            elif disagg_sim:
                verdict = self._smoothagent_admission.should_admit_be(
                    candidate,
                    candidate_chunk,
                    snapshot,
                    mode="disaggregated",
                    t_budget_ms=t_budget_ms_const,
                )
            elif colo_tbt:
                ok, reason, pred_with, _marginal = self._colo_tbt_slack_gate(
                    candidate,
                    candidate_chunk,
                    snapshot,
                    pending_be_reqs=pending_be_this_tick,
                    in_flight_be_reqs=bg_cost_reqs_before_continuation,
                    in_flight_be_cost_state=bg_cost_state_before_continuation,
                    decode_baseline_ms=colo_decode_baseline,
                    lc_marginal_ms=colo_lc_marginal,
                )
                from sglang.srt.managers.smoothagent.admission_controller import (
                    AdmissionVerdict as _Verdict,
                )
                verdict = _Verdict(
                    admit=ok,
                    reason=reason,
                    predicted_latency_ms=pred_with,
                    t_budget_ms=None,
                )
            else:
                verdict = self._smoothagent_admission.should_admit_be(
                    candidate,
                    candidate_chunk,
                    snapshot,
                    mode="colocated",
                    tbt_slo_ms=self._smoothagent_admission.request_slo_tbt_ms(candidate),
                )
            self._smoothagent_last_predicted_latency_ms = float(
                verdict.predicted_latency_ms
            )
            if verdict.t_budget_ms is not None and math.isfinite(verdict.t_budget_ms):
                self._smoothagent_last_t_budget_ms = float(verdict.t_budget_ms)

            if not verdict.admit:
                self._smoothagent_paper_denied_total += 1
                self._bg_stats.bg_denied_total += 1
                logger.debug(
                    "Paper gate denied BE chunk: rid=%s reason=%s "
                    "predicted=%.2fms tbt_bound=%.2fms",
                    candidate.rid, verdict.reason,
                    verdict.predicted_latency_ms,
                    verdict.t_budget_ms or -1.0,
                )
                break  # The latency model is monotonic: deny now → deny others.

            # Try to add via adder. If add_one_req returns OTHER (budget
            # exhausted) or NO_TOKEN, stop the admission loop.
            self.bg_waiting_queue.pop(0)
            self._anti_starvation.on_dequeue(candidate.rid)
            prev_can_run_len = len(adder.can_run_list)
            adder.add_one_req(
                candidate,
                has_chunked_req=bool(self.chunked_reqs)
                or bool(self.bg_chunked_reqs),
                truncation_align_size=truncation_align_size,
                chunk_cap=c_chunk,
            )
            bg_added = (
                len(adder.can_run_list) > prev_can_run_len
                and adder.can_run_list[-1] is candidate
            )
            if not bg_added:
                # Adder rejected (budget exhausted, no memory, etc.); put
                # the candidate back at the queue head and stop admitting.
                # Subsequent BE candidates would face the same exhaustion.
                self.bg_waiting_queue.insert(0, candidate)
                self._anti_starvation.on_enqueue(candidate.rid)
                break

            self._smoothagent_paper_admitted_total += 1
            self._bg_stats.bg_admitted_total += 1
            tokens_added = candidate.extend_input_len
            self._bg_stats.bg_prefill_tokens_total += tokens_added
            self._bg_stats.bg_prefill_tokens_last_tick = tokens_added
            # Track this admission so subsequent iterations see its cost.
            if (per_lc_budget and in_flight_lcs) or colo_tbt:
                pending_be_this_tick.append(candidate)

            # If add_one_req emitted candidate as a fresh chunked req,
            # transfer it from adder.new_chunked_reqs to bg_chunked_reqs
            # so scheduler.py:absorb-loop doesn't double-count it as FG.
            if candidate in adder.new_chunked_reqs:
                adder.new_chunked_reqs.remove(candidate)
                self.bg_chunked_reqs.append(candidate)
                scheduled_bg_chunked_reqs.append(candidate)
                # Mirror FG: bump is_chunked so output processor takes
                # the chunked branch. BG has max_new_tokens=0, so the
                # non-chunked branch would set FINISH_LENGTH immediately.
                candidate.is_chunked += 1
                candidate.lookahead_pending_more_chunks = True
                logger.info(
                    "BG chunked prefill: rid=%s, tokens=%d, continuing",
                    candidate.rid, tokens_added,
                )
            else:
                candidate.lookahead_pending_more_chunks = False
                logger.info(
                    "BG prefill complete: rid=%s, tokens=%d",
                    candidate.rid, tokens_added,
                )
        return scheduled_bg_chunked_reqs

    def _maybe_append_bg_prefill_legacy(self, adder):
        """
        Legacy single-admit path. Preserves SLOBudgetController + the
        original ``gate_bg_request`` flow for deployments that haven't
        enabled paper-mode. Single in-flight bg chunked req at a time.
        """
        free_blocks = self.token_to_kv_pool_allocator.available_size()
        total_blocks = self.token_to_kv_pool_allocator.size
        fg_queue_len = len(self.waiting_queue)

        decision, _t_budget, c_chunk, _m_free = self._slo_budget_ctrl.decide(
            self._latency_predictor.predicted_ttft_ms,
            fg_queue_len,
            free_blocks,
            total_blocks,
        )

        if decision == BGDecision.DENY:
            bg_rids = [r.rid for r in self.bg_waiting_queue]
            force, starving_rid = self._anti_starvation.should_force_allow(bg_rids)
            if force:
                decision = BGDecision.ALLOW_LIMITED
                c_chunk = self._lookahead_config.bg_max_chunk_tokens // 4
                self._bg_stats.bg_force_allowed_total += 1
                logger.info(
                    "Anti-starvation: force-allowing BG, starving_rid=%s",
                    starving_rid,
                )

        if decision in (BGDecision.DENY, BGDecision.SKIP):
            return []

        if self._latency_predictor.tbt_sample_count > 0:
            c_chunk = self._chunk_mixer.compute_chunk_size(
                self._latency_predictor.predicted_tbt_ms,
                base_chunk=c_chunk,
            )

        c_chunk = self._clamp_bg_chunk_under_fg_pressure(c_chunk, fg_queue_len)
        if c_chunk <= 0:
            return []
        scheduled_bg_chunked_reqs: List[Req] = []

        bg_req = self.bg_chunked_req
        popped_from_queue = False
        if bg_req is not None:
            session = self.sessions.get(bg_req.session_id)
            if session is not None:
                self._sync_pending_lookahead_req_from_branch(session, bg_req)
            if not self.gate_bg_request(bg_req):
                return []
            bg_req.init_next_round_input(self.tree_cache)
        elif len(self.bg_waiting_queue) > 0:
            candidate = self.bg_waiting_queue[0]
            session = self.sessions.get(candidate.session_id)
            if session is not None:
                self._sync_pending_lookahead_req_from_branch(session, candidate)
            if not self.gate_bg_request(candidate):
                return []
            bg_req = self.bg_waiting_queue.pop(0)
            popped_from_queue = True
            self._anti_starvation.on_dequeue(bg_req.rid)
            bg_req.init_next_round_input(self.tree_cache)

        if bg_req is None:
            return []
        if not self._has_bg_req_slot_capacity(adder):
            if popped_from_queue:
                self.bg_waiting_queue.insert(0, bg_req)
                self._anti_starvation.on_enqueue(bg_req.rid)
            return []

        prev_can_run_len = len(adder.can_run_list)
        prev_rem_input_tokens = adder.rem_input_tokens
        prev_rem_chunk_tokens = adder.rem_chunk_tokens

        adder.rem_input_tokens = min(adder.rem_input_tokens, c_chunk)
        if adder.rem_chunk_tokens is None:
            adder.rem_chunk_tokens = c_chunk
        else:
            adder.rem_chunk_tokens = min(adder.rem_chunk_tokens, c_chunk)

        adder.add_one_req(
            bg_req,
            has_chunked_req=bool(self.chunked_reqs)
            or bool(self.bg_chunked_reqs),
            truncation_align_size=getattr(self, 'truncation_align_size', 1),
        )

        bg_added = (
            len(adder.can_run_list) > prev_can_run_len
            and adder.can_run_list[-1] is bg_req
        )

        if bg_added:
            consumed_input_tokens = adder.ceil_paged_tokens(bg_req.extend_input_len)
            tokens_added = bg_req.extend_input_len
            adder.rem_input_tokens = prev_rem_input_tokens - consumed_input_tokens
            if prev_rem_chunk_tokens is None:
                adder.rem_chunk_tokens = None
            else:
                adder.rem_chunk_tokens = prev_rem_chunk_tokens - consumed_input_tokens
            self._bg_stats.bg_prefill_tokens_total += tokens_added
            self._bg_stats.bg_prefill_tokens_last_tick = tokens_added

            if bg_req in adder.new_chunked_reqs:
                adder.new_chunked_reqs.remove(bg_req)
                self.bg_chunked_reqs = [bg_req]
                scheduled_bg_chunked_reqs.append(bg_req)
                bg_req.is_chunked += 1
                bg_req.lookahead_pending_more_chunks = True
                logger.info(
                    "BG chunked prefill: rid=%s, tokens=%d, continuing",
                    bg_req.rid, tokens_added,
                )
            else:
                self.bg_chunked_reqs = []
                bg_req.lookahead_pending_more_chunks = False
                logger.info(
                    "BG prefill complete: rid=%s, tokens=%d",
                    bg_req.rid, tokens_added,
                )
        else:
            adder.rem_input_tokens = prev_rem_input_tokens
            adder.rem_chunk_tokens = prev_rem_chunk_tokens
            if not self.bg_chunked_reqs:
                self.bg_waiting_queue.insert(0, bg_req)
                self._anti_starvation.on_enqueue(bg_req.rid)
        return scheduled_bg_chunked_reqs

    # ── BG decode gate ───────────────────────────────────────────────

    def restore_bg_decode_reqs(self):
        """
        Restore BG decode reqs that were paused in the previous tick.
        Must be called at the start of get_next_batch_to_run, before
        update_running_batch, so paused BG reqs rejoin the running_batch.

        CORRECTNESS NOTE: filter_bg_decode_reqs removed paused reqs from
        every per-position tensor via ``batch.filter_batch``. Simply
        extending ``running_batch.reqs`` would leave ``running_batch.reqs``
        larger than its tensors, so we requeue the paused reqs through the
        normal BG waiting queue instead. Their KV-cache positions were
        kept allocated (filter_batch does not free KV cache), so the
        scheduler's prefill path will see them as cache-hit reqs and
        resume without re-tokenizing from scratch.
        """
        if not hasattr(self, '_bg_decode_paused_reqs'):
            return
        if self._bg_decode_paused_reqs:
            paused = self._bg_decode_paused_reqs
            self._bg_decode_paused_reqs = []
            # Push back to the BG waiting queue so the scheduler resumes
            # them via normal prefill admission. This avoids the stale-
            # tensor corruption that the previous `reqs.extend` path
            # caused.
            if hasattr(self, 'bg_waiting_queue'):
                for req in paused:
                    self.bg_waiting_queue.insert(0, req)
            logger.debug(
                "Restored %d paused BG decode reqs (requeued to bg_waiting_queue)",
                len(paused),
            )

    def filter_bg_decode_reqs(self, batch):
        """
        Gate BG decode requests. If FG has pressure, BG decode reqs
        are paused for this tick (saved to _bg_decode_paused_reqs,
        restored next tick via restore_bg_decode_reqs).

        CORRECTNESS NOTE: we must use ``batch.filter_batch(keep_indices=...)``
        to keep every per-position tensor (req_pool_indices, seq_lens,
        sampling_info, output_ids tensor, ...) consistent with ``batch.reqs``.
        Previously this method only reassigned ``batch.reqs = remaining``,
        which left stale tensor rows for the paused BG positions. The
        subsequent forward pass then produced ``N+K`` next_token_ids while
        output processing iterated over the trimmed ``N`` reqs via
        ``zip(batch.reqs, next_token_ids)`` — mis-aligning tokens to reqs
        and causing BG summary generations to ingest tokens from other
        positions (observed as character/token-level interleaving in the
        summary text).
        """
        if not hasattr(self, '_slo_budget_ctrl'):
            return

        fg_queue_len = len(self.waiting_queue)
        allow_bg_decode = self._slo_budget_ctrl.should_allow_bg_decode(
            self._latency_predictor.predicted_ttft_ms,
            fg_queue_len,
        )

        if allow_bg_decode:
            return

        keep_indices: List[int] = []
        paused: List[Req] = []
        for idx, req in enumerate(batch.reqs):
            if (
                req.is_bg
                and len(req.output_ids) > 0
                and not getattr(req, "lookahead_promoted", False)
            ):
                paused.append(req)
            else:
                keep_indices.append(idx)

        if not paused:
            return

        self._bg_decode_paused_reqs = paused
        self._bg_stats.bg_decode_skipped_total += len(paused)
        batch.filter_batch(keep_indices=keep_indices)
        logger.debug(
            "BG decode paused: %d reqs, fg_queue=%d",
            len(paused), fg_queue_len,
        )

    # ── FG / BG lifecycle tracking ───────────────────────────────────

    def on_fg_tick(self, req: Req):
        """Update latency predictor when a FG request produces a token or finishes."""
        if not hasattr(self, '_latency_predictor'):
            return

        if hasattr(req, 'time_stats') and req.time_stats is not None:
            ttft = getattr(req.time_stats, 'prefill_latency', None)
            if ttft is not None:
                ttft_ms = ttft * 1000
                self._latency_predictor.update_ttft(ttft_ms)

                # Check for burst spike
                self._burst_guard.check_spike(
                    ttft_ms, self._latency_predictor.predicted_ttft_ms
                )

                # Update drift detector
                self._drift_detector.update(ttft_ms)

    def on_bg_request_complete(self, req: Req):
        """Track BG request completion."""
        if getattr(req, "lookahead_complete_handled", False):
            return

        def finish_handled() -> None:
            req.lookahead_complete_handled = True
            req.lookahead_pending_more_chunks = False
            if hasattr(self, '_bg_stats'):
                self._bg_stats.bg_completed_total += 1
            self.bg_chunked_reqs = [
                r for r in self.bg_chunked_reqs if r is not req
            ]
            if self.promoted_lookahead_req is req:
                self.promoted_lookahead_req = None

        if (
            req.reduce_request_type != ReduceRequestType.LOOKAHEAD_REDUCE
            or req.session_id is None
        ):
            finish_handled()
            return
        session = self.sessions.get(req.session_id)
        if session is None:
            finish_handled()
            return
        if getattr(req.finished_reason, "is_error", False):
            session.clear_lookahead(req.lookahead_task_id or None, self.tree_cache)
            finish_handled()
            return
        active_task_id = session.get_pending_lookahead_task()
        active_branch = session.get_lookahead_branch(task_id=req.lookahead_task_id)
        if active_branch is not None:
            active_task_id = active_branch.task_id
        if active_task_id is None:
            active_task_id = session.lookahead_commit_task_id
        if active_task_id != req.lookahead_task_id:
            logger.info(
                "Dropping stale lookahead completion: session=%s task=%s active_task=%s",
                req.session_id,
                req.lookahead_task_id,
                active_task_id,
            )
            session.clear_lookahead(req.lookahead_task_id or None, self.tree_cache)
            finish_handled()
            return
        if active_branch is not None and (
            active_branch.source_ctx_len_tokens != req.lookahead_source_ctx_len_tokens
        ):
            logger.info(
                "Lookahead branch advanced beyond completed reduce: session=%s task=%s "
                "version=%d completed_source_ctx=%d active_source_ctx=%d; resuming latest frontier",
                req.session_id,
                req.lookahead_task_id,
                req.lookahead_version,
                req.lookahead_source_ctx_len_tokens,
                active_branch.source_ctx_len_tokens,
            )
            self._resume_advanced_lookahead_branch(session, active_branch)
            finish_handled()
            return
        if (
            active_branch is None
            and session.lookahead_pending_version == req.lookahead_version
            and session.lookahead_pending_ctx_len != 0
            and session.lookahead_pending_ctx_len != req.lookahead_source_ctx_len_tokens
        ):
            logger.info(
                "Dropping stale lookahead completion with ctx mismatch: session=%s task=%s "
                "pending_ctx=%s completed_ctx=%s",
                req.session_id,
                req.lookahead_task_id,
                session.lookahead_pending_ctx_len,
                req.lookahead_source_ctx_len_tokens,
            )
            session.clear_lookahead(req.lookahead_task_id or None, self.tree_cache)
            finish_handled()
            return
        if (
            session.lookahead_pending_task_id is not None
            and req.lookahead_task_id != session.lookahead_pending_task_id
        ):
            logger.info(
                "Dropping stale lookahead completion with pending task mismatch: "
                "session=%s completed_task=%s pending_task=%s",
                req.session_id,
                req.lookahead_task_id,
                session.lookahead_pending_task_id,
            )
            session.clear_lookahead(req.lookahead_task_id or None, self.tree_cache)
            finish_handled()
            return
        if (
            req.lookahead_strategy == "summarize"
            and req.lookahead_artifact.get("stage") == "summary_generation"
        ):
            finish_handled()
            self._on_summarize_generation_complete(session, req)
            return

        match_result = self.tree_cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids=req.origin_input_ids,
                    extra_key=req.extra_key,
                )
            )
        )
        if len(match_result.device_indices) == 0:
            logger.warning(
                "Discarding lookahead result without reusable device prefix: session=%s task=%s",
                req.session_id,
                req.lookahead_task_id,
            )
            session.clear_lookahead(req.lookahead_task_id or None, self.tree_cache)
            finish_handled()
            return

        session.attach_lookahead_result(
            LookaheadResult(
                task_id=req.lookahead_task_id,
                strategy=req.lookahead_strategy,
                artifact=dict(req.lookahead_artifact),
                reduced_input_ids=list(req.origin_input_ids),
                reduced_kv_indices=match_result.device_indices.clone(),
                prefix_indices=match_result.device_indices.clone(),
                last_device_node=match_result.last_device_node,
                last_host_node=match_result.last_host_node,
                host_hit_length=match_result.host_hit_length,
                cache_protected_len=len(match_result.device_indices),
                source_ctx_len_tokens=req.lookahead_source_ctx_len_tokens,
                version=session.conversation_version,
                is_ready=True,
            ),
            self.tree_cache,
        )
        logger.info(
            "Lookahead result ready: session=%s version=%d source_ctx=%d reduced_ctx=%d committed_kv=%d",
            req.session_id,
            req.lookahead_version,
            req.lookahead_source_ctx_len_tokens,
            len(req.origin_input_ids),
            len(match_result.device_indices),
        )
        finish_handled()

    def on_bg_request_tick(self, req: Req):
        """Track BG decode progress (for token accounting)."""
        pass  # Token accounting is done in track_batch_token_stats

    def handle_lookahead_control(
        self,
        recv_req: LookaheadControlReqInput,
    ) -> LookaheadControlReqOutput:
        if recv_req.action == "warmup":
            return self._submit_lookahead_warmup(recv_req)
        if recv_req.action == "append":
            return self._append_lookahead(recv_req)
        if recv_req.action == "summary_ready":
            return self._summarize_external_ready(recv_req)
        if recv_req.action == "commit":
            return self._commit_lookahead(recv_req)
        if recv_req.action == "clear":
            return self._clear_lookahead(recv_req)
        return LookaheadControlReqOutput(
            session_id=recv_req.session_id,
            action=recv_req.action,
            task_id=recv_req.task_id,
            success=False,
            message=f"Unknown lookahead action: {recv_req.action}",
        )

    def _submit_lookahead_warmup(
        self,
        recv_req: LookaheadControlReqInput,
    ) -> LookaheadControlReqOutput:
        session = self.sessions.get(recv_req.session_id)
        if session is None:
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=False,
                message=f"Session {recv_req.session_id} does not exist.",
            )

        control = dict(recv_req.control or {})
        source_ctx_len_tokens = int(control.get("usage", len(recv_req.input_ids or [])))
        pending = (
            session.lookahead_pending_version == session.conversation_version
            and session.lookahead_pending_ctx_len == source_ctx_len_tokens
            and session.lookahead_pending_task_id == recv_req.task_id
        )
        main_state = LookaheadMainState(
            ctx_len_tokens=source_ctx_len_tokens,
            capacity=self.max_req_input_len,
            soft_threshold=int(control.get("soft_limit", session.soft_threshold)),
            hard_threshold=int(control.get("hard_limit", session.hard_threshold)),
        )
        la_state = LookaheadState(
            version=session.conversation_version,
            pending=pending,
            task_id=recv_req.task_id,
            strategy=recv_req.strategy,
            control=control,
        )
        trigger_allowed = (recv_req.strategy == "sub_agent" and not pending) or should_trigger(main_state, la_state)
        if not trigger_allowed:
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=True,
                accepted=False,
                message="Soft threshold not reached or identical warmup already pending.",
            )
        if recv_req.strategy == "summarize":
            control = self._summarize_control_with_frontier_metadata(
                {
                    **dict(control),
                    "tail_text": str(control.get("tail_text") or ""),
                    "usage": source_ctx_len_tokens,
                }
            )
            if (
                session.lookahead_pending_task_id is not None
                and session.lookahead_pending_task_id != recv_req.task_id
            ):
                session.clear_lookahead(tree_cache=self.tree_cache)
            if bool(control.get("defer_summary_generation")):
                session.mark_lookahead_pending(recv_req.task_id or "", source_ctx_len_tokens)
                session.set_lookahead_commit_control(
                    recv_req.task_id or "",
                    control,
                )
                logger.info(
                    "Accepted deferred summarize warmup: session=%s task=%s version=%d source_ctx=%d",
                    recv_req.session_id,
                    recv_req.task_id,
                    session.conversation_version,
                    source_ctx_len_tokens,
                )
                return LookaheadControlReqOutput(
                    session_id=recv_req.session_id,
                    action=recv_req.action,
                    task_id=recv_req.task_id,
                    success=True,
                    accepted=True,
                    message="Summarize generation deferred to external PD decode path.",
                )
            bg_req = self._build_summarize_generation_req(
                recv_req=recv_req,
                version=session.conversation_version,
                source_ctx_len_tokens=source_ctx_len_tokens,
                control=control,
            )
            if bg_req is None:
                return LookaheadControlReqOutput(
                    session_id=recv_req.session_id,
                    action=recv_req.action,
                    task_id=recv_req.task_id,
                    success=True,
                    accepted=False,
                    message="Summarize warmup could not build a summary prompt.",
                )
            session.mark_lookahead_pending(recv_req.task_id or "", source_ctx_len_tokens)
            session.set_lookahead_commit_control(
                recv_req.task_id or "",
                control,
            )
            bg_req.time_stats.set_wait_queue_entry_time()
            self.bg_waiting_queue.append(bg_req)
            self._anti_starvation.on_enqueue(bg_req.rid)
            logger.info(
                "Accepted summarize warmup: session=%s task=%s version=%d source_ctx=%d",
                recv_req.session_id,
                recv_req.task_id,
                session.conversation_version,
                source_ctx_len_tokens,
            )
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=True,
                accepted=True,
            )

        reduced_input_ids, artifact = transform((recv_req.input_ids or [], control), la_state)
        if recv_req.strategy == "sub_agent":
            messages = control.get("messages")
            if (
                isinstance(messages, list)
                and messages
                and self.tokenizer is not None
                and hasattr(self.tokenizer, "apply_chat_template")
            ):
                # Pass tools through when caller provided them so the chat
                # template renders the tool-schema block (Qwen / Llama-3.1+ /
                # Mistral-tool-use). Without this, the warmup prefix omits the
                # tool block while the real /generate request includes it,
                # breaking the prefix match from the very first token.
                tools = control.get("tools")
                template_kwargs = {
                    "tokenize": True,
                    "add_generation_prompt": False,
                }
                if isinstance(tools, list) and tools:
                    template_kwargs["tools"] = tools
                try:
                    templated_ids = list(
                        self.tokenizer.apply_chat_template(messages, **template_kwargs)
                    )
                except Exception as exc:
                    logger.warning(
                        "sub_agent apply_chat_template failed (%s); falling back to transform passthrough.",
                        exc,
                    )
                else:
                    if templated_ids:
                        reduced_input_ids = templated_ids
                        if isinstance(artifact, dict):
                            artifact["chat_template_applied"] = True
                            if "tools" in template_kwargs:
                                artifact["tools_applied"] = True
        if not reduced_input_ids:
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=True,
                accepted=False,
                message="Lookahead warmup has no candidate prefix.",
            )

        if recv_req.strategy in {"sliding_window", "offloading", "sub_agent"}:
            active_branch = session.get_lookahead_branch()
            if active_branch is not None and (
                active_branch.task_id != (recv_req.task_id or "")
                or active_branch.strategy != recv_req.strategy
            ):
                session.clear_lookahead(tree_cache=self.tree_cache)
                self._drop_local_lookahead_task(recv_req.session_id)
            session.start_lookahead_branch(
                task_id=recv_req.task_id or "",
                strategy=recv_req.strategy,
                input_ids=reduced_input_ids,
                input_text=recv_req.text,
                source_ctx_len_tokens=source_ctx_len_tokens,
                artifact=artifact,
            )

        if (
            session.lookahead_pending_task_id is not None
            and session.lookahead_pending_task_id != recv_req.task_id
        ):
            session.clear_lookahead(tree_cache=self.tree_cache)
            self._drop_local_lookahead_task(recv_req.session_id)
        bg_req = self._build_lookahead_reduce_req(
            session_id=recv_req.session_id,
            task_id=recv_req.task_id or "",
            strategy=recv_req.strategy,
            input_text=recv_req.text,
            reduced_input_ids=reduced_input_ids,
            version=session.conversation_version,
            source_ctx_len_tokens=source_ctx_len_tokens,
            artifact=artifact,
        )
        session.mark_lookahead_pending(recv_req.task_id or "", source_ctx_len_tokens)
        bg_req.time_stats.set_wait_queue_entry_time()
        self.bg_waiting_queue.append(bg_req)
        self._anti_starvation.on_enqueue(bg_req.rid)
        logger.info(
            "Accepted lookahead warmup: session=%s task=%s version=%d source_ctx=%d reduced_ctx=%d",
            recv_req.session_id,
            recv_req.task_id,
            session.conversation_version,
            source_ctx_len_tokens,
            len(reduced_input_ids),
        )
        return LookaheadControlReqOutput(
            session_id=recv_req.session_id,
            action=recv_req.action,
            task_id=recv_req.task_id,
            success=True,
            accepted=True,
        )

    def _accept_branch_append(
        self,
        session,
        recv_req: LookaheadControlReqInput,
        branch,
        source_ctx_len_tokens: int,
    ) -> LookaheadControlReqOutput:
        session.refresh_lookahead_pending(branch.task_id, source_ctx_len_tokens)
        # Summarize artifacts depend on the exact tail text digest, so an
        # append invalidates the previous ready summary. Token-preserving
        # strategies can still reuse the latest ready prefix and prefill the
        # newly appended tail on the foreground path.
        if branch.strategy == "summarize":
            session.clear_lookahead_result(branch.task_id, self.tree_cache)
        local_req = self._find_mutable_local_lookahead_req(
            recv_req.session_id,
            branch.task_id,
        )
        if local_req is not None:
            if not self._sync_lookahead_req_to_branch(local_req, branch):
                self._drop_local_lookahead_task(recv_req.session_id, branch.task_id)
                local_req = None
            else:
                promotion_message = ""
                if session.lookahead_commit_task_id == branch.task_id:
                    promotion_state = self.promote_lookahead_for_commit(
                        session,
                        branch.task_id,
                    )
                    promotion_message = (
                        "Appended lookahead branch promoted onto the hard-commit fast lane."
                        if promotion_state == "promoted"
                        else (
                            "Appended lookahead branch queued behind an active hard-commit fast lane."
                            if promotion_state == "queued"
                            else (
                                "Appended lookahead branch is already running on the hard-commit path."
                                if promotion_state == "running"
                                else ""
                            )
                        )
                    )
                logger.info(
                    "Accepted in-place lookahead append: session=%s task=%s version=%d source_ctx=%d reduced_ctx=%d",
                    recv_req.session_id,
                    branch.task_id,
                    session.conversation_version,
                    source_ctx_len_tokens,
                    len(branch.input_ids),
                )
                return LookaheadControlReqOutput(
                    session_id=recv_req.session_id,
                    action=recv_req.action,
                    task_id=branch.task_id,
                    success=True,
                    accepted=True,
                    message=promotion_message,
                )

        inflight_req = self._find_inflight_lookahead_req(
            recv_req.session_id,
            branch.task_id,
            session.conversation_version,
        )
        if inflight_req is not None:
            logger.info(
                "Accepted latched lookahead append: session=%s task=%s version=%d source_ctx=%d reduced_ctx=%d",
                recv_req.session_id,
                branch.task_id,
                session.conversation_version,
                source_ctx_len_tokens,
                len(branch.input_ids),
            )
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=branch.task_id,
                success=True,
                accepted=True,
                message=(
                    "Append latched on an active lookahead branch; the next chunk boundary will continue from the updated frontier."
                ),
            )

        self._drop_local_lookahead_task(recv_req.session_id, branch.task_id)
        bg_req = self._build_lookahead_reduce_req(
            session_id=recv_req.session_id,
            task_id=branch.task_id,
            strategy=recv_req.strategy,
            input_text=branch.input_text,
            reduced_input_ids=list(branch.input_ids),
            version=session.conversation_version,
            source_ctx_len_tokens=source_ctx_len_tokens,
            artifact=dict(branch.artifact),
        )
        bg_req.time_stats.set_wait_queue_entry_time()
        if session.lookahead_commit_task_id == branch.task_id and self.promoted_lookahead_req is None:
            self.promoted_lookahead_req = bg_req
            self._mark_promoted_lookahead_req(bg_req)
            promotion_message = "Appended lookahead branch promoted onto the hard-commit fast lane."
        elif session.lookahead_commit_task_id == branch.task_id:
            self._mark_promoted_lookahead_req(bg_req)
            self.waiting_queue.append(bg_req)
            promotion_message = (
                "Appended lookahead branch queued behind an active hard-commit fast lane."
            )
        else:
            self.bg_waiting_queue.append(bg_req)
            self._anti_starvation.on_enqueue(bg_req.rid)
            promotion_message = ""

        logger.info(
            "Accepted lookahead append: session=%s task=%s version=%d source_ctx=%d reduced_ctx=%d",
            recv_req.session_id,
            branch.task_id,
            session.conversation_version,
            source_ctx_len_tokens,
            len(branch.input_ids),
        )
        return LookaheadControlReqOutput(
            session_id=recv_req.session_id,
            action=recv_req.action,
            task_id=branch.task_id,
            success=True,
            accepted=True,
            message=promotion_message,
        )

    def _resume_advanced_lookahead_branch(
        self,
        session,
        branch,
    ) -> None:
        """Continue a branch that advanced while an older reduce req was still in flight."""
        session.refresh_lookahead_pending(
            branch.task_id,
            branch.source_ctx_len_tokens,
        )

        local_req = self._find_mutable_local_lookahead_req(
            session.session_id,
            branch.task_id,
        )
        if local_req is not None:
            if self._sync_lookahead_req_to_branch(local_req, branch):
                if session.lookahead_commit_task_id == branch.task_id:
                    self.promote_lookahead_for_commit(session, branch.task_id)
                return
            self._drop_local_lookahead_task(session.session_id, branch.task_id)

        inflight_req = self._find_inflight_lookahead_req(
            session.session_id,
            branch.task_id,
            session.conversation_version,
            branch.source_ctx_len_tokens,
        )
        if inflight_req is not None:
            return

        bg_req = self._build_lookahead_reduce_req(
            session_id=session.session_id,
            task_id=branch.task_id,
            strategy=branch.strategy,
            input_text=branch.input_text,
            reduced_input_ids=list(branch.input_ids),
            version=session.conversation_version,
            source_ctx_len_tokens=branch.source_ctx_len_tokens,
            artifact=dict(branch.artifact),
        )
        bg_req.time_stats.set_wait_queue_entry_time()
        if (
            session.lookahead_commit_task_id == branch.task_id
            and self.promoted_lookahead_req is None
        ):
            self.promoted_lookahead_req = bg_req
            self._mark_promoted_lookahead_req(bg_req)
            return
        if session.lookahead_commit_task_id == branch.task_id:
            self._mark_promoted_lookahead_req(bg_req)
            self.waiting_queue.append(bg_req)
            return

        self.bg_waiting_queue.append(bg_req)
        self._anti_starvation.on_enqueue(bg_req.rid)

    def _append_lookahead(
        self,
        recv_req: LookaheadControlReqInput,
    ) -> LookaheadControlReqOutput:
        session = self.sessions.get(recv_req.session_id)
        if session is None:
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=False,
                message=f"Session {recv_req.session_id} does not exist.",
            )

        if recv_req.strategy not in {"sliding_window", "offloading", "summarize", "sub_agent"}:
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=True,
                accepted=False,
                message=f"Append is not supported for strategy {recv_req.strategy}.",
            )

        control = dict(recv_req.control or {})
        source_ctx_len_tokens = int(
            control.get(
                "usage",
                session.lookahead_pending_ctx_len
                or (
                    session.get_lookahead_branch(task_id=recv_req.task_id).source_ctx_len_tokens
                    if session.get_lookahead_branch(task_id=recv_req.task_id) is not None
                    else 0
                ),
            )
        )
        if recv_req.strategy == "summarize":
            merged_control = dict(
                session.get_lookahead_commit_control(recv_req.task_id) or {}
            )
            merged_control.update(control)
            merged_control["tail_text"] = str(merged_control.get("tail_text") or "")
            merged_control["usage"] = source_ctx_len_tokens
            merged_control = self._summarize_control_with_frontier_metadata(
                merged_control
            )
            session.set_lookahead_commit_control(recv_req.task_id or "", merged_control)

            branch = session.get_lookahead_branch(
                task_id=recv_req.task_id or "",
                strategy="summarize",
            )
            if branch is not None:
                # Default OFF: matches plain /generate. Opt in
                # (use_chat_template=True) for /v1/chat/completions clients
                # whose tokens go through apply_chat_template server-side.
                use_chat_template = bool(merged_control.get("use_chat_template", False))
                tail_messages = merged_control.get("tail_messages")

                if (
                    use_chat_template
                    and tail_messages
                    and self.tokenizer is not None
                    and hasattr(self.tokenizer, "apply_chat_template")
                ):
                    # Rebuild full message list: [summary] + tail_messages
                    summary_text = str(
                        branch.artifact.get("summary_text", "")
                    ).strip()
                    if not summary_text:
                        return LookaheadControlReqOutput(
                            session_id=recv_req.session_id,
                            action=recv_req.action,
                            task_id=recv_req.task_id,
                            success=True,
                            accepted=False,
                            message="Summarize branch has no summary_text for chat template append.",
                        )
                    # summary_text is already formatted by
                    # _on_summarize_generation_complete; use directly.
                    # Build complete post-commit prompt:
                    # overhead_prefix + summary + overhead_suffix + tail
                    messages: list = []
                    overhead_prefix = merged_control.get("overhead_prefix")
                    if overhead_prefix:
                        messages.extend(overhead_prefix)
                    messages.append(
                        {"role": "system", "content": summary_text}
                    )
                    overhead_suffix = merged_control.get("overhead_suffix")
                    if overhead_suffix:
                        messages.extend(overhead_suffix)
                    messages.extend(tail_messages)
                    tools = merged_control.get("tools")
                    template_kwargs = {
                        "tokenize": True,
                        "add_generation_prompt": False,
                    }
                    if isinstance(tools, list) and tools:
                        template_kwargs["tools"] = tools
                    full_input_ids = list(
                        self.tokenizer.apply_chat_template(messages, **template_kwargs)
                    )
                    # Diff against existing branch tokens
                    existing_len = len(branch.input_ids)
                    if (
                        existing_len <= len(full_input_ids)
                        and full_input_ids[:existing_len]
                        == branch.input_ids[:existing_len]
                    ):
                        delta_input_ids = full_input_ids[existing_len:]
                        if not delta_input_ids:
                            return LookaheadControlReqOutput(
                                session_id=recv_req.session_id,
                                action=recv_req.action,
                                task_id=recv_req.task_id,
                                success=True,
                                accepted=True,
                                message="Summarize chat template append: no new tokens.",
                            )
                    else:
                        logger.warning(
                            "Summarize chat template append prefix mismatch: "
                            "session=%s task=%s existing_len=%d new_len=%d",
                            recv_req.session_id,
                            recv_req.task_id,
                            existing_len,
                            len(full_input_ids),
                        )
                        return LookaheadControlReqOutput(
                            session_id=recv_req.session_id,
                            action=recv_req.action,
                            task_id=recv_req.task_id,
                            success=True,
                            accepted=False,
                            message="Summarize chat template append prefix mismatch.",
                        )

                    branch = session.append_lookahead_branch(
                        task_id=recv_req.task_id or "",
                        strategy=recv_req.strategy,
                        delta_input_ids=delta_input_ids,
                        delta_text=None,
                        source_ctx_len_tokens=source_ctx_len_tokens,
                        artifact_update=dict(merged_control),
                    )
                else:
                    summary_text = str(
                        branch.artifact.get("summary_text", "")
                    ).strip()
                    if summary_text:
                        reduced_text = self._build_summary_prefill_text(
                            summary_text,
                            str(merged_control.get("tail_text") or ""),
                        )
                        full_input_ids = self._encode_lookahead_text(reduced_text)
                        existing_len = len(branch.input_ids)
                        if (
                            existing_len <= len(full_input_ids)
                            and full_input_ids[:existing_len] == branch.input_ids
                        ):
                            branch = session.append_lookahead_branch(
                                task_id=recv_req.task_id or "",
                                strategy=recv_req.strategy,
                                delta_input_ids=full_input_ids[existing_len:],
                                delta_text=None,
                                source_ctx_len_tokens=source_ctx_len_tokens,
                                artifact_update=dict(merged_control),
                            )
                            if branch is not None:
                                branch.input_text = reduced_text
                        else:
                            branch.input_ids = list(full_input_ids)
                            branch.input_text = reduced_text
                            branch.source_ctx_len_tokens = source_ctx_len_tokens
                            branch.artifact.update(dict(merged_control))
                    else:
                        branch = session.append_lookahead_branch(
                            task_id=recv_req.task_id or "",
                            strategy=recv_req.strategy,
                            delta_input_ids=list(recv_req.input_ids or []),
                            delta_text=recv_req.text,
                            source_ctx_len_tokens=source_ctx_len_tokens,
                            artifact_update=dict(merged_control),
                        )

                if branch is None:
                    return LookaheadControlReqOutput(
                        session_id=recv_req.session_id,
                        action=recv_req.action,
                        task_id=recv_req.task_id,
                        success=True,
                        accepted=False,
                        message="No active summarize branch exists for append.",
                    )
                return self._accept_branch_append(
                    session,
                    recv_req,
                    branch,
                    source_ctx_len_tokens,
                )

            warmup_artifact = session.get_lookahead_warmup_artifact(recv_req.task_id)
            if warmup_artifact is not None:
                promotion_state = self._start_summarize_prefill(
                    session=session,
                    task_id=recv_req.task_id or "",
                    control=merged_control,
                    warmup_artifact=warmup_artifact,
                    promote=session.lookahead_commit_task_id == (recv_req.task_id or ""),
                )
                if promotion_state != "missing":
                    return LookaheadControlReqOutput(
                        session_id=recv_req.session_id,
                        action=recv_req.action,
                        task_id=recv_req.task_id,
                        success=True,
                        accepted=True,
                        message=(
                            "Summarize append started or refreshed the second-stage prefill branch."
                        ),
                    )

            local_req = self._find_mutable_local_lookahead_req(
                recv_req.session_id,
                recv_req.task_id or "",
            )
            inflight_req = self._find_inflight_lookahead_req(
                recv_req.session_id,
                recv_req.task_id or "",
                session.conversation_version,
            )
            if (
                session.lookahead_pending_task_id == (recv_req.task_id or "")
                or local_req is not None
                or inflight_req is not None
            ):
                return LookaheadControlReqOutput(
                    session_id=recv_req.session_id,
                    action=recv_req.action,
                    task_id=recv_req.task_id,
                    success=True,
                    accepted=True,
                    message=(
                        "Summarize append latched on the pending summary-generation stage; second-stage prefill will use the updated tail frontier."
                    ),
                )

            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=True,
                accepted=False,
                message="No active summarize branch or pending summarize generation exists for append.",
            )

        full_input_ids = list(recv_req.input_ids or [])
        delta_input_ids = full_input_ids
        delta_text = recv_req.text

        if control.get("full_messages") and full_input_ids:
            existing_branch = session.get_lookahead_branch(
                task_id=recv_req.task_id or "",
                strategy=recv_req.strategy,
            )
            if existing_branch is not None:
                existing_len = len(existing_branch.input_ids)
                if (
                    existing_len <= len(full_input_ids)
                    and full_input_ids[:existing_len] == existing_branch.input_ids
                ):
                    delta_input_ids = full_input_ids[existing_len:]
                    delta_text = None
                    if not delta_input_ids:
                        return LookaheadControlReqOutput(
                            session_id=recv_req.session_id,
                            action=recv_req.action,
                            task_id=recv_req.task_id,
                            success=True,
                            accepted=True,
                            message="No new tokens to append (full_messages prefix matches branch).",
                        )
                else:
                    logger.warning(
                        "full_messages prefix mismatch: session=%s task=%s "
                        "branch_len=%d full_len=%d",
                        recv_req.session_id,
                        recv_req.task_id,
                        existing_len,
                        len(full_input_ids),
                    )
                    return LookaheadControlReqOutput(
                        session_id=recv_req.session_id,
                        action=recv_req.action,
                        task_id=recv_req.task_id,
                        success=True,
                        accepted=False,
                        message="full_messages token prefix does not match existing branch.",
                    )

        branch = session.append_lookahead_branch(
            task_id=recv_req.task_id or "",
            strategy=recv_req.strategy,
            delta_input_ids=delta_input_ids,
            delta_text=delta_text,
            source_ctx_len_tokens=source_ctx_len_tokens,
            artifact_update=dict(control),
        )
        if branch is None:
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=True,
                accepted=False,
                message="No active lookahead branch exists for append.",
            )
        return self._accept_branch_append(
            session,
            recv_req,
            branch,
            source_ctx_len_tokens,
        )

    def _commit_lookahead(
        self,
        recv_req: LookaheadControlReqInput,
    ) -> LookaheadControlReqOutput:
        session = self.sessions.get(recv_req.session_id)
        if session is None:
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=False,
                message=f"Session {recv_req.session_id} does not exist.",
            )

        control = dict(recv_req.control or {})
        main_state = LookaheadMainState(
            ctx_len_tokens=int(
                control.get(
                    "usage",
                    session.lookahead_pending_ctx_len
                    or (
                        session.lookahead_result.source_ctx_len_tokens
                        if session.lookahead_result is not None
                        else 0
                    ),
                )
            ),
            capacity=self.max_req_input_len,
            soft_threshold=int(control.get("soft_limit", session.soft_threshold)),
            hard_threshold=int(control.get("hard_limit", session.hard_threshold)),
        )
        if not should_commit(main_state):
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=True,
                accepted=False,
                message="Hard threshold not reached.",
            )
        if recv_req.strategy == "summarize":
            merged_control = dict(
                session.get_lookahead_commit_control(recv_req.task_id or "") or {}
            )
            merged_control.update(control)
            session.set_lookahead_commit_control(
                recv_req.task_id or "",
                self._summarize_control_with_frontier_metadata(
                    {
                        **merged_control,
                        "tail_text": self._decode_lookahead_text(
                            recv_req.text,
                            recv_req.input_ids,
                        ),
                        "usage": main_state.ctx_len_tokens,
                    }
                ),
            )
            control = dict(
                session.get_lookahead_commit_control(recv_req.task_id or "") or control
            )

        commit_source_ctx_len = (
            main_state.ctx_len_tokens if recv_req.strategy == "summarize" else None
        )
        stale_summarize_artifact: Optional[Dict[str, object]] = None
        result = session.request_lookahead_commit(
            recv_req.task_id,
            source_ctx_len_tokens=commit_source_ctx_len,
        )
        if (
            recv_req.strategy == "summarize"
            and result is not None
            and not self._summarize_result_matches_control(result, control)
        ):
            stale_summarize_artifact = dict(result.artifact)
            logger.info(
                "Discarding stale summarize lookahead result: session=%s task=%s "
                "expected_tail_digest=%s actual_tail_digest=%s",
                recv_req.session_id,
                result.task_id,
                self._summarize_frontier_digest(control),
                result.artifact.get("summary_tail_digest"),
            )
            session.clear_lookahead_result(result.task_id, self.tree_cache)
            session.lookahead_commit_task_id = result.task_id
            result = None
        if result is None:
            target_task_id = session.lookahead_commit_task_id
            if (
                target_task_id is None
                or (
                    recv_req.task_id is not None
                    and target_task_id != recv_req.task_id
                )
            ):
                return LookaheadControlReqOutput(
                    session_id=recv_req.session_id,
                    action=recv_req.action,
                    task_id=recv_req.task_id,
                    success=True,
                    accepted=False,
                    ready=False,
                    message="Lookahead warmup is not ready.",
                )
            if recv_req.strategy == "summarize":
                warmup_artifact = session.get_lookahead_warmup_artifact(target_task_id)
                if (
                    warmup_artifact is None
                    and stale_summarize_artifact is not None
                    and str(stale_summarize_artifact.get("summary_text") or "").strip()
                ):
                    warmup_artifact = stale_summarize_artifact
                if warmup_artifact is not None:
                    if bool(control.get("allow_local_fallback")):
                        fallback_artifact = self._build_summarize_local_fallback_artifact(
                            task_id=target_task_id,
                            warmup_artifact=warmup_artifact,
                        )
                        if fallback_artifact is not None:
                            session.clear_lookahead(target_task_id, self.tree_cache)
                            self._drop_local_lookahead_task(
                                recv_req.session_id,
                                target_task_id,
                            )
                            return LookaheadControlReqOutput(
                                session_id=recv_req.session_id,
                                action=recv_req.action,
                                task_id=target_task_id,
                                success=True,
                                accepted=True,
                                ready=True,
                                artifact={
                                    "task_id": target_task_id,
                                    "strategy": "summarize",
                                    "metadata": fallback_artifact,
                                },
                            )
                    existing_state = self.promote_lookahead_for_commit(
                        session,
                        target_task_id,
                    )
                    if existing_state != "missing":
                        return LookaheadControlReqOutput(
                            session_id=recv_req.session_id,
                            action=recv_req.action,
                            task_id=target_task_id,
                            success=True,
                            accepted=True,
                            ready=False,
                            message=(
                                "Summarize prefill promoted to the hard-commit fast lane."
                                if existing_state == "promoted"
                                else (
                                    "Summarize prefill queued behind an active hard-commit fast lane."
                                    if existing_state == "queued"
                                    else "Summarize prefill is already running; commit will finalize on ready."
                                )
                            ),
                        )
                    promotion_state = self._start_summarize_prefill(
                        session=session,
                        task_id=target_task_id,
                        control=session.get_lookahead_commit_control(target_task_id) or {},
                        warmup_artifact=warmup_artifact,
                        promote=True,
                    )
                    if promotion_state != "missing":
                        return LookaheadControlReqOutput(
                            session_id=recv_req.session_id,
                            action=recv_req.action,
                            task_id=target_task_id,
                            success=True,
                            accepted=True,
                            ready=False,
                            message=(
                                "Summarize prefill promoted to the hard-commit fast lane."
                                if promotion_state == "promoted"
                                else (
                                    "Summarize prefill queued behind an active hard-commit fast lane."
                                    if promotion_state == "queued"
                                    else "Summarize prefill is already running; commit will finalize on ready."
                                )
                            ),
                        )

            promotion_state = self.promote_lookahead_for_commit(session, target_task_id)
            if promotion_state == "missing":
                return LookaheadControlReqOutput(
                    session_id=recv_req.session_id,
                    action=recv_req.action,
                    task_id=recv_req.task_id,
                    success=True,
                    accepted=False,
                    ready=False,
                    message="Lookahead warmup is not ready.",
                )

            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=target_task_id,
                success=True,
                accepted=True,
                ready=False,
                message=(
                    "Lookahead warmup promoted to the hard-commit fast lane."
                    if promotion_state == "promoted"
                    else (
                        "Lookahead warmup queued behind an active hard-commit fast lane."
                        if promotion_state == "queued"
                        else "Lookahead warmup is already running; commit will finalize on ready."
                    )
                ),
            )

        return LookaheadControlReqOutput(
            session_id=recv_req.session_id,
            action=recv_req.action,
            task_id=result.task_id,
            success=True,
            accepted=True,
            ready=True,
            artifact={
                "task_id": result.task_id,
                "strategy": result.strategy,
                "metadata": dict(result.artifact),
            },
        )

    def _summarize_external_ready(
        self,
        recv_req: LookaheadControlReqInput,
    ) -> LookaheadControlReqOutput:
        session = self.sessions.get(recv_req.session_id)
        if session is None:
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=False,
                message=f"Session {recv_req.session_id} does not exist.",
            )
        task_id = recv_req.task_id or ""
        if not task_id:
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=False,
                message="summary_ready requires task_id.",
            )

        control = dict(recv_req.control or {})
        summary_text = str(control.get("summary_text") or "").strip()
        if not summary_text:
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=task_id,
                success=True,
                accepted=False,
                message="summary_ready missing summary_text.",
            )

        artifact = dict(control)
        artifact["stage"] = "summary_generation_external"
        artifact["summary_text"] = self._format_summary_message_content(
            summary_text,
            artifact,
        )
        artifact["external_summary_generation"] = True
        session.set_lookahead_warmup_artifact(task_id, artifact)

        commit_control = session.get_lookahead_commit_control(task_id)
        if commit_control is None:
            logger.info(
                "External summarize artifact ready without commit control: session=%s task=%s",
                recv_req.session_id,
                task_id,
            )
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=task_id,
                success=True,
                accepted=True,
                ready=False,
                message="External summarize artifact stored.",
            )

        promotion_state = self._start_summarize_prefill(
            session=session,
            task_id=task_id,
            control=commit_control,
            warmup_artifact=artifact,
            promote=session.lookahead_commit_task_id == task_id,
        )
        if promotion_state == "missing":
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=task_id,
                success=True,
                accepted=False,
                ready=False,
                message="External summarize artifact could not start prefill.",
            )

        logger.info(
            "External summarize warmup ready: session=%s task=%s state=%s",
            recv_req.session_id,
            task_id,
            promotion_state,
        )
        return LookaheadControlReqOutput(
            session_id=recv_req.session_id,
            action=recv_req.action,
            task_id=task_id,
            success=True,
            accepted=True,
            ready=False,
            message=(
                "External summarize prefill promoted."
                if promotion_state == "promoted"
                else (
                    "External summarize prefill queued."
                    if promotion_state == "queued"
                    else "External summarize prefill already running."
                )
            ),
        )

    def _clear_lookahead(
        self,
        recv_req: LookaheadControlReqInput,
    ) -> LookaheadControlReqOutput:
        session = self.sessions.get(recv_req.session_id)
        if session is None:
            return LookaheadControlReqOutput(
                session_id=recv_req.session_id,
                action=recv_req.action,
                task_id=recv_req.task_id,
                success=False,
                message=f"Session {recv_req.session_id} does not exist.",
            )
        session.clear_lookahead(recv_req.task_id, self.tree_cache)
        self._drop_local_lookahead_task(recv_req.session_id, recv_req.task_id)
        return LookaheadControlReqOutput(
            session_id=recv_req.session_id,
            action=recv_req.action,
            task_id=recv_req.task_id,
            success=True,
            accepted=True,
        )

    def _decode_lookahead_text(
        self,
        text: Optional[str],
        input_ids: Optional[List[int]],
    ) -> str:
        if text is not None:
            return text
        if not input_ids or self.tokenizer is None:
            return ""
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)

    def _encode_lookahead_text(self, text: str) -> List[int]:
        if self.tokenizer is None or not text:
            return []
        encoded = self.tokenizer(text)
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            return list(input_ids[0])
        return list(input_ids)

    _SUMMARY_SYSTEM_PROMPT = (
        "You are a precise conversation summarizer for an AI agent's working memory. "
        "You receive a ReAct-style transcript (tool calls and observations) and must "
        "produce a faithful, compact summary. Output ONLY the summary prose — no "
        "preface, no headings, no bullet points, no code fences."
    )

    _SUMMARY_INSTRUCTION_SUFFIX = (
        "\n\n---\n"
        "Task: Summarize the transcript above in 3-6 sentences. "
        "Preserve: user's goal, files/functions inspected or edited, decisions made, "
        "tool outputs that influenced next steps, open tasks. "
        "Omit: verbatim tool output, repeated boilerplate, ReAct scaffolding. "
        "Output only the summary text."
    )

    def _build_summary_prompt(self, source_text: str) -> str:
        # Instruction placed AFTER source so it's closest to the generation
        # position — autoregressive attention weighs recent tokens more heavily.
        return f"{source_text}{self._SUMMARY_INSTRUCTION_SUFFIX}"

    def _encode_summary_prompt(self, source_text: str) -> List[int]:
        """Encode the summarize prompt using chat template for instruct models."""
        if self.tokenizer is None:
            return []
        user_content = self._build_summary_prompt(source_text)
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": self._SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            try:
                return list(
                    self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True
                    )
                )
            except Exception:
                pass
        return self._encode_lookahead_text(
            f"{self._SUMMARY_SYSTEM_PROMPT}\n\n{user_content}"
        )

    def _format_summary_message_content(
        self,
        summary_text: str,
        control: Dict[str, Any],
    ) -> str:
        prefix = str(control.get("summary_prefix", "Context summary:\n"))
        cleaned_summary = summary_text.strip()
        if not prefix:
            return cleaned_summary
        return f"{prefix}{cleaned_summary}".strip()

    def _build_summary_prefill_text(
        self,
        summary_text: str,
        tail_text: str,
    ) -> str:
        if tail_text:
            return f"System: {summary_text}\n{tail_text}"
        return f"System: {summary_text}"

    @staticmethod
    def _summarize_frontier_digest(control: Dict[str, object]) -> str:
        payload = {
            "tail_text": str(control.get("tail_text") or ""),
            "tail_messages": control.get("tail_messages"),
            "overhead_prefix": control.get("overhead_prefix"),
            "overhead_suffix": control.get("overhead_suffix"),
            "tools": control.get("tools"),
            "use_chat_template": bool(control.get("use_chat_template", False)),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _summarize_control_with_frontier_metadata(
        self,
        control: Dict[str, object],
    ) -> Dict[str, object]:
        normalized = dict(control)
        tail_text = str(normalized.get("tail_text") or "")
        normalized["tail_text"] = tail_text
        normalized["summary_tail_digest"] = self._summarize_frontier_digest(normalized)
        normalized["summary_tail_text_len"] = len(tail_text)
        tail_messages = normalized.get("tail_messages")
        if isinstance(tail_messages, list):
            normalized["summary_tail_message_count"] = len(tail_messages)
        return normalized

    def _summarize_result_matches_control(
        self,
        result: Optional[LookaheadResult],
        control: Dict[str, object],
    ) -> bool:
        if result is None or result.strategy != "summarize":
            return True
        expected = self._summarize_frontier_digest(control)
        actual = result.artifact.get("summary_tail_digest")
        return isinstance(actual, str) and actual == expected

    def _build_summarize_local_fallback_artifact(
        self,
        *,
        task_id: str,
        warmup_artifact: Dict[str, object],
    ) -> Optional[Dict[str, object]]:
        summary_text = str(warmup_artifact.get("summary_text", "")).strip()
        replace_count = int(warmup_artifact.get("replace_count", 0))
        if not summary_text or replace_count <= 0:
            logger.warning(
                "Summarize local fallback skipped: task=%s "
                "summary_text_len=%d replace_count=%d (need both > 0)",
                task_id,
                len(summary_text),
                replace_count,
            )
            return None
        return {
            "task_id": task_id,
            "strategy": "summarize",
            "replace_count": replace_count,
            "summary_text": summary_text,
            "local_fallback": True,
        }

    def _build_summarize_generation_req(
        self,
        recv_req: LookaheadControlReqInput,
        version: int,
        source_ctx_len_tokens: int,
        control: Dict[str, Any],
    ) -> Optional[Req]:
        source_text = self._decode_lookahead_text(recv_req.text, recv_req.input_ids)
        if not source_text:
            logger.warning(
                "Summarize generation skipped: empty source_text after decode. "
                "session=%s task=%s text_len=%d input_ids_len=%d",
                recv_req.session_id,
                recv_req.task_id,
                len(recv_req.text or ""),
                len(recv_req.input_ids or []),
            )
            return None
        summary_prompt = self._build_summary_prompt(source_text)
        summary_prompt_ids = self._encode_summary_prompt(source_text)
        if not summary_prompt_ids:
            logger.warning(
                "Summarize generation skipped: encode_summary_prompt returned "
                "empty. session=%s task=%s source_text_len=%d prompt_len=%d",
                recv_req.session_id,
                recv_req.task_id,
                len(source_text),
                len(summary_prompt or ""),
            )
            return None
        summary_max_tokens = int(control.get("summary_max_tokens", 256))
        sampling_params = SamplingParams(
            max_new_tokens=max(1, summary_max_tokens),
            temperature=0.0,
        )
        artifact = dict(control)
        artifact["stage"] = "summary_generation"
        artifact["source_text"] = source_text
        return self._build_lookahead_reduce_req(
            session_id=recv_req.session_id,
            task_id=recv_req.task_id or "",
            strategy=recv_req.strategy,
            input_text=summary_prompt,
            reduced_input_ids=summary_prompt_ids,
            version=version,
            source_ctx_len_tokens=source_ctx_len_tokens,
            artifact=artifact,
            sampling_params=sampling_params,
        )

    def _build_summarize_prefill_req(
        self,
        session,
        task_id: str,
        control: Dict[str, object],
        warmup_artifact: Dict[str, object],
    ) -> Optional[Req]:
        control = self._summarize_control_with_frontier_metadata(control)
        summary_text = str(warmup_artifact.get("summary_text", "")).strip()
        if not summary_text:
            logger.warning(
                "Summarize prefill skipped: empty summary_text "
                "(generation stage produced nothing). session=%s task=%s",
                session.session_id,
                task_id,
            )
            return None
        tail_text = str(control.get("tail_text") or "").strip()
        reduced_text = self._build_summary_prefill_text(summary_text, tail_text)
        # Default OFF: /generate (plain text endpoint) tokenizes via plain
        # tokenizer.encode (tokenizer_manager._tokenize_texts), so plain
        # encode of `f"System: {summary}\n{tail}"` matches /generate exactly.
        # Clients posting through /v1/chat/completions (which DOES call
        # apply_chat_template on the server) should opt in by setting
        # control["use_chat_template"]=True so the radix prefix matches the
        # templated tokens of the real request.
        use_chat_template = bool(control.get("use_chat_template", False))
        if (
            use_chat_template
            and self.tokenizer is not None
            and hasattr(self.tokenizer, "apply_chat_template")
        ):
            # summary_text is already formatted by _on_summarize_generation_complete;
            # use it directly as the system message content.
            # Build complete post-commit prompt: overhead_prefix + summary + overhead_suffix + tail
            messages: list = []
            overhead_prefix = control.get("overhead_prefix")
            if overhead_prefix:
                messages.extend(overhead_prefix)
            messages.append({"role": "system", "content": summary_text})
            overhead_suffix = control.get("overhead_suffix")
            if overhead_suffix:
                messages.extend(overhead_suffix)
            tail_messages = control.get("tail_messages")
            if tail_messages:
                messages.extend(tail_messages)
            # Pass tools through when present so chat-template-rendered tool
            # blocks match between warmup and real /generate.
            tools = control.get("tools")
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": False,
            }
            if isinstance(tools, list) and tools:
                template_kwargs["tools"] = tools
            reduced_input_ids = list(
                self.tokenizer.apply_chat_template(messages, **template_kwargs)
            )
        else:
            reduced_input_ids = self._encode_lookahead_text(reduced_text)
        if not reduced_input_ids:
            logger.warning(
                "Summarize prefill skipped: tokenization yielded empty token "
                "list. session=%s task=%s use_chat_template=%s "
                "summary_text_len=%d tail_text_len=%d",
                session.session_id,
                task_id,
                use_chat_template,
                len(summary_text),
                len(tail_text),
            )
            return None
        source_ctx_len_tokens = int(
            control.get("usage") or session.lookahead_pending_ctx_len or len(reduced_input_ids)
        )
        artifact = dict(warmup_artifact)
        artifact.update(
            {
                "tail_text": tail_text,
                "summary_tail_digest": control["summary_tail_digest"],
                "summary_tail_text_len": control["summary_tail_text_len"],
            }
        )
        if "summary_tail_message_count" in control:
            artifact["summary_tail_message_count"] = control[
                "summary_tail_message_count"
            ]
        artifact["stage"] = "summary_prefill"
        return self._build_lookahead_reduce_req(
            session_id=session.session_id,
            task_id=task_id,
            strategy="summarize",
            input_text=reduced_text,
            reduced_input_ids=reduced_input_ids,
            version=session.conversation_version,
            source_ctx_len_tokens=source_ctx_len_tokens,
            artifact=artifact,
        )

    def _start_summarize_prefill(
        self,
        session,
        task_id: str,
        control: Dict[str, object],
        warmup_artifact: Dict[str, object],
        *,
        promote: bool,
    ) -> str:
        bg_req = self._build_summarize_prefill_req(session, task_id, control, warmup_artifact)
        if bg_req is None:
            return "missing"
        source_ctx_len_tokens = int(
            control.get("usage") or session.lookahead_pending_ctx_len or len(bg_req.origin_input_ids)
        )
        session.start_lookahead_branch(
            task_id=task_id,
            strategy="summarize",
            input_ids=list(bg_req.origin_input_ids),
            input_text=bg_req.origin_input_text,
            source_ctx_len_tokens=source_ctx_len_tokens,
            artifact=dict(bg_req.lookahead_artifact),
        )
        session.refresh_lookahead_pending(task_id, source_ctx_len_tokens)
        bg_req.time_stats.set_wait_queue_entry_time()
        if promote and self.promoted_lookahead_req is None:
            self.promoted_lookahead_req = bg_req
            self._mark_promoted_lookahead_req(bg_req)
            return "promoted"
        if promote and self.promoted_lookahead_req is not None:
            self._mark_promoted_lookahead_req(bg_req)
            self.waiting_queue.append(bg_req)
            return "queued"
        self.bg_waiting_queue.append(bg_req)
        self._anti_starvation.on_enqueue(bg_req.rid)
        return "queued"

    def _on_summarize_generation_complete(self, session, req: Req) -> None:
        summary_text = self._decode_lookahead_text(None, req.output_ids_through_stop).strip()
        if not summary_text:
            logger.warning(
                "Discarding empty summarize artifact: session=%s task=%s",
                req.session_id,
                req.lookahead_task_id,
            )
            session.clear_lookahead(req.lookahead_task_id or None, self.tree_cache)
            return
        artifact = dict(req.lookahead_artifact)
        artifact["summary_text"] = self._format_summary_message_content(
            summary_text,
            artifact,
        )
        session.set_lookahead_warmup_artifact(req.lookahead_task_id, artifact)
        logger.info(
            "Summarize warmup ready: session=%s version=%d task=%s",
            req.session_id,
            req.lookahead_version,
            req.lookahead_task_id,
        )
        commit_control = session.get_lookahead_commit_control(req.lookahead_task_id)
        if commit_control is None:
            return
        promotion_state = self._start_summarize_prefill(
            session=session,
            task_id=req.lookahead_task_id,
            control=commit_control,
            warmup_artifact=artifact,
            promote=session.lookahead_commit_task_id == req.lookahead_task_id,
        )
        if promotion_state == "missing":
            logger.warning(
                "Summarize prefill missing after generation completed; "
                "clearing lookahead. session=%s task=%s "
                "(see prior 'Summarize prefill skipped' log for cause)",
                req.session_id,
                req.lookahead_task_id,
            )
            session.clear_lookahead(req.lookahead_task_id or None, self.tree_cache)

    def _build_lookahead_reduce_req(
        self,
        session_id: str,
        task_id: str,
        strategy: str,
        input_text: Optional[str],
        reduced_input_ids: List[int],
        version: int,
        source_ctx_len_tokens: int,
        artifact: Dict[str, Any],
        sampling_params: Optional[SamplingParams] = None,
    ) -> Req:
        if sampling_params is None:
            sampling_params = SamplingParams(max_new_tokens=0)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.model_config.vocab_size)

        bg_req = Req(
            rid=f"la:{task_id}:{uuid.uuid4().hex[:8]}",
            origin_input_text=input_text,
            origin_input_ids=list(reduced_input_ids),
            origin_input_ids_unpadded=tuple(reduced_input_ids),
            sampling_params=sampling_params,
            session_id=session_id,
            vocab_size=self.model_config.vocab_size,
            priority=self._get_lowest_lookahead_priority(),
            metrics_collector=self.metrics_collector if self.enable_metrics else None,
        )
        bg_req.tokenizer = self.tokenizer
        bg_req.request_channel = RequestChannel.BACKGROUND
        bg_req.reduce_request_type = ReduceRequestType.LOOKAHEAD_REDUCE
        bg_req.lookahead_version = version
        bg_req.lookahead_source_ctx_len_tokens = source_ctx_len_tokens
        bg_req.lookahead_task_id = task_id
        bg_req.lookahead_strategy = strategy
        bg_req.lookahead_artifact = dict(artifact)
        bg_req.ctx_len_tokens = source_ctx_len_tokens
        return bg_req

    # ── Per-batch token stats ────────────────────────────────────────

    def track_batch_token_stats(self, batch):
        """
        Per-rid token statistics. Called after process_batch_result().
        Breaks down prefill/decode tokens by FG/BG.
        """
        if not hasattr(self, '_bg_stats'):
            return

        fg_prefill = 0
        bg_prefill = 0
        fg_decode = 0
        bg_decode = 0
        fg_prefill_reqs = 0
        bg_prefill_reqs = 0
        fg_prefixes: List[int] = []
        bg_prefixes: List[int] = []
        decoding_reqs = set(getattr(batch, "decoding_reqs", None) or [])

        for req in batch.reqs:
            if hasattr(batch, 'forward_mode') and batch.forward_mode.is_extend():
                if req in decoding_reqs:
                    if req.is_bg:
                        bg_decode += 1
                    else:
                        fg_decode += 1
                    continue
                tokens = req.extend_input_len if hasattr(req, 'extend_input_len') else 0
                if req.is_bg:
                    bg_prefill += tokens
                    if tokens > 0:
                        bg_prefill_reqs += 1
                        bg_prefixes.append(
                            max(0, int(_bl_estimate_prefix(req)) - int(tokens))
                        )
                else:
                    fg_prefill += tokens
                    if tokens > 0:
                        fg_prefill_reqs += 1
                        fg_prefixes.append(
                            max(0, int(_bl_estimate_prefix(req)) - int(tokens))
                        )
            elif hasattr(batch, 'forward_mode') and batch.forward_mode.is_decode():
                if req.is_bg:
                    bg_decode += 1
                else:
                    fg_decode += 1

        tick_ts_unix = time.time()
        prev_tick_ts_unix = float(getattr(self._bg_stats, "last_tick_ts_unix", 0.0) or 0.0)
        tick_gap_ms = (
            max(0.0, (tick_ts_unix - prev_tick_ts_unix) * 1000.0)
            if prev_tick_ts_unix > 0.0
            else 0.0
        )
        self._bg_stats.last_tick_seq += 1
        self._bg_stats.last_tick_ts_unix = tick_ts_unix
        self._bg_stats.last_tick_gap_ms = tick_gap_ms
        host_forward_elapsed_ms = float(
            getattr(batch, "lookahead_forward_elapsed_ms", 0.0) or 0.0
        )
        forward_elapsed_ms = host_forward_elapsed_ms
        forward_elapsed_source = "host"
        start_event = getattr(batch, "lookahead_forward_start_event", None)
        end_event = getattr(batch, "lookahead_forward_end_event", None)
        if start_event is not None and end_event is not None:
            try:
                end_event.synchronize()
                cuda_elapsed_ms = float(start_event.elapsed_time(end_event))
                if cuda_elapsed_ms > 1e-6:
                    forward_elapsed_ms = cuda_elapsed_ms
                    forward_elapsed_source = "cuda_event"
                elif host_forward_elapsed_ms > 0.0:
                    forward_elapsed_source = "host_cuda_event_zero"
                else:
                    forward_elapsed_source = "cuda_event_zero"
            except Exception:
                if host_forward_elapsed_ms > 0.0:
                    forward_elapsed_source = "host_cuda_event_error"
                else:
                    forward_elapsed_source = "cuda_event_error"
        self._bg_stats.last_forward_elapsed_ms = forward_elapsed_ms
        self._bg_stats.last_forward_elapsed_source = forward_elapsed_source
        self._bg_stats.fg_prefill_tokens_last_tick = fg_prefill
        self._bg_stats.bg_prefill_tokens_last_tick = bg_prefill
        self._bg_stats.fg_decode_tokens_last_tick = fg_decode
        self._bg_stats.bg_decode_tokens_last_tick = bg_decode
        self._bg_stats.fg_prefill_reqs_last_tick = fg_prefill_reqs
        self._bg_stats.bg_prefill_reqs_last_tick = bg_prefill_reqs
        self._bg_stats.fg_prefill_prefix_avg_last_tick = (
            round(float(sum(fg_prefixes)) / float(len(fg_prefixes)), 3)
            if fg_prefixes
            else 0.0
        )
        self._bg_stats.bg_prefill_prefix_avg_last_tick = (
            round(float(sum(bg_prefixes)) / float(len(bg_prefixes)), 3)
            if bg_prefixes
            else 0.0
        )
        self._bg_stats.fg_prefill_prefix_max_last_tick = (
            max(fg_prefixes) if fg_prefixes else 0
        )
        self._bg_stats.bg_prefill_prefix_max_last_tick = (
            max(bg_prefixes) if bg_prefixes else 0
        )
        self._bg_stats.bg_prefill_active = (bg_prefill > 0)

        tick_record = {
            "last_tick_seq": int(self._bg_stats.last_tick_seq),
            "last_tick_ts_unix": float(self._bg_stats.last_tick_ts_unix),
            "last_tick_gap_ms": float(self._bg_stats.last_tick_gap_ms),
            "last_forward_elapsed_ms": float(
                self._bg_stats.last_forward_elapsed_ms
            ),
            "last_forward_elapsed_source": str(
                self._bg_stats.last_forward_elapsed_source
            ),
            "fg_prefill_tokens_last_tick": int(fg_prefill),
            "bg_prefill_tokens_last_tick": int(bg_prefill),
            "fg_decode_tokens_last_tick": int(fg_decode),
            "bg_decode_tokens_last_tick": int(bg_decode),
            "fg_prefill_reqs_last_tick": int(fg_prefill_reqs),
            "bg_prefill_reqs_last_tick": int(bg_prefill_reqs),
            "fg_prefill_prefix_avg_last_tick": float(
                self._bg_stats.fg_prefill_prefix_avg_last_tick
            ),
            "bg_prefill_prefix_avg_last_tick": float(
                self._bg_stats.bg_prefill_prefix_avg_last_tick
            ),
            "fg_prefill_prefix_max_last_tick": int(
                self._bg_stats.fg_prefill_prefix_max_last_tick
            ),
            "bg_prefill_prefix_max_last_tick": int(
                self._bg_stats.bg_prefill_prefix_max_last_tick
            ),
            "bg_prefill_active": bool(self._bg_stats.bg_prefill_active),
        }
        self._bg_stats.tick_history.append(tick_record)
        max_history = int(getattr(self._bg_stats, "tick_history_max_len", 256) or 256)
        if len(self._bg_stats.tick_history) > max_history:
            del self._bg_stats.tick_history[:-max_history]

        if bg_decode > 0:
            self._bg_stats.bg_decode_tokens_total += bg_decode

    # ── ReduceSignal attachment ──────────────────────────────────────

    def _attach_reduce_signals(self, reqs: List[Req]):
        """
        For each request with a session, check SoftHardDetector and
        attach a _reduce_signal attribute if threshold is crossed.
        """
        if not hasattr(self, '_soft_hard_detector'):
            return

        for req in reqs:
            session_id = getattr(req, 'session_id', None)
            if session_id is None:
                continue

            ctx_len = len(req.origin_input_ids) + len(req.output_ids)
            capacity = getattr(self, 'max_req_input_len', 0)
            if capacity <= 0:
                continue

            signal = self._soft_hard_detector.detect(ctx_len, capacity, session_id)
            if signal.level != ReduceSignalLevel.NONE:
                req._reduce_signal = signal
                logger.debug(
                    "ReduceSignal attached: rid=%s, level=%s, ratio=%.2f",
                    req.rid, signal.level.value, signal.ctx_ratio,
                )

    # ── Stats endpoint ───────────────────────────────────────────────

    def get_bg_scheduling_stats(self) -> Dict[str, Any]:
        """Return JSON-serializable stats for /bg_scheduling/stats endpoint."""
        if not hasattr(self, '_bg_stats'):
            return {"enabled": False}

        def req_debug(req: Optional[Req]) -> Optional[Dict[str, Any]]:
            if req is None:
                return None
            def safe_len(value: Any) -> int:
                if value is None:
                    return 0
                try:
                    return len(value)
                except Exception:
                    return 0

            return {
                "rid": getattr(req, "rid", None),
                "task_id": getattr(req, "lookahead_task_id", None),
                "pending_more": bool(getattr(req, "lookahead_pending_more_chunks", False)),
                "promoted": bool(getattr(req, "lookahead_promoted", False)),
                "handled": bool(getattr(req, "lookahead_complete_handled", False)),
                "finished": getattr(req, "finished_reason", None) is not None,
                "is_chunked": int(getattr(req, "is_chunked", 0) or 0),
                "req_pool_idx": getattr(req, "req_pool_idx", None),
                "extend_input_len": int(getattr(req, "extend_input_len", 0) or 0),
                "prefix_len": safe_len(getattr(req, "prefix_indices", None)),
                "origin_len": safe_len(getattr(req, "origin_input_ids", None)),
            }

        stats = self._bg_stats.to_dict()
        stats["enabled"] = True
        stats["latency_predictor"] = self._latency_predictor.to_dict()
        stats["drift_detector"] = self._drift_detector.to_dict()
        stats["burst_guard"] = self._burst_guard.to_dict()
        stats["oldest_bg_wait_s"] = round(
            self._anti_starvation.get_oldest_wait_s(
                [r.rid for r in self.bg_waiting_queue]
            ), 2
        )
        # SmoothAgent paper-mode telemetry — additive, only meaningful when
        # SGLANG_SMOOTHAGENT_PAPER_MODE=1, but always present for stable schema.
        try:
            predicted_ms = self.predict_batch_latency_ms()
        except Exception:
            predicted_ms = 0.0
        stats["smoothagent_paper_mode"] = bool(
            getattr(self, "_smoothagent_paper_mode", False)
        )
        stats["smoothagent_paper_admitted_total"] = int(
            getattr(self, "_smoothagent_paper_admitted_total", 0)
        )
        stats["smoothagent_paper_denied_total"] = int(
            getattr(self, "_smoothagent_paper_denied_total", 0)
        )
        stats["smoothagent_paper_predicted_batch_latency_ms"] = round(
            float(predicted_ms), 3
        )
        stats["smoothagent_paper_last_admission_predicted_latency_ms"] = round(
            float(getattr(self, "_smoothagent_last_predicted_latency_ms", 0.0)), 3
        )
        stats["smoothagent_paper_last_t_budget_ms"] = round(
            float(getattr(self, "_smoothagent_last_t_budget_ms", 0.0)), 3
        )
        stats["smoothagent_lc_prefill_admitted_total"] = int(
            getattr(self, "_smoothagent_lc_prefill_admitted_total", 0)
        )
        stats["smoothagent_lc_prefill_deferred_total"] = int(
            getattr(self, "_smoothagent_lc_prefill_deferred_total", 0)
        )
        stats["smoothagent_lc_prefill_passive_skips_total"] = int(
            getattr(self, "_smoothagent_lc_prefill_passive_skips_total", 0)
        )
        stats["smoothagent_paper_disaggregated_prefill"] = (
            self._is_paper_disaggregated_prefill()
            if hasattr(self, "_smoothagent_admission") else False
        )
        running_batch = getattr(self, "running_batch", None)
        last_batch = getattr(self, "last_batch", None)
        stats["debug_waiting_queue_len"] = len(getattr(self, "waiting_queue", []) or [])
        stats["debug_bg_waiting_queue_len"] = len(
            getattr(self, "bg_waiting_queue", []) or []
        )
        stats["debug_bg_chunked_reqs_len"] = len(
            getattr(self, "bg_chunked_reqs", []) or []
        )
        stats["debug_chunked_reqs_len"] = len(getattr(self, "chunked_reqs", []) or [])
        stats["debug_running_batch_size"] = (
            running_batch.batch_size() if running_batch is not None else 0
        )
        stats["debug_running_batch_full"] = (
            bool(getattr(running_batch, "batch_is_full", False))
            if running_batch is not None
            else False
        )
        stats["debug_last_batch_size"] = (
            last_batch.batch_size() if last_batch is not None else 0
        )
        stats["debug_promoted"] = req_debug(
            getattr(self, "promoted_lookahead_req", None)
        )
        stats["debug_waiting_promoted"] = [
            req_debug(req)
            for req in (getattr(self, "waiting_queue", []) or [])
            if getattr(req, "lookahead_promoted", False)
        ][:8]
        stats["debug_bg_waiting_head"] = [
            req_debug(req) for req in (getattr(self, "bg_waiting_queue", []) or [])[:8]
        ]
        return stats
