"""
Lookahead BG Scheduling — Controller Collection.

8 stateless component classes, each with a single responsibility:
  1. LatencyPredictor      — EMA-based FG latency tracking with burst normalization
  2. SLOBudgetController   — Core decision: ALLOW / ALLOW_LIMITED / DENY / SKIP
  3. ChunkMixer            — TBT-guided BG chunk sizing
  4. SoftHardDetector       — ctx_len / capacity => soft/hard ReduceSignal
  5. AntiStarvationGuard   — Force-allow BG after max wait time
  6. DriftDetector         — Detect FG latency trend shifts
  7. ExtremeBurstGuard     — Extreme spike protection
  8. BGSchedulingStats     — Aggregated statistics (JSON schema)
"""

import collections
import dataclasses
import enum
import math
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

from sglang.srt.managers.lookahead_config import LookaheadConfig


# ── 1. LatencyPredictor ─────────────────────────────────────────────────


class LatencyPredictor:
    """
    EMA-based latency predictor for FG requests.
    Tracks both TTFT (prefill latency) and TBT (time-between-tokens).
    Supports burst normalization: ignores spikes > burst_multiplier * ema.
    """

    def __init__(self, alpha: float = 0.1, burst_multiplier: float = 3.0,
                 initial_ttft_ema_ms: float = 0.0):
        self._alpha = alpha
        self._burst_multiplier = burst_multiplier

        self._ttft_ema_ms: float = initial_ttft_ema_ms
        self._tbt_ema_ms: float = 0.0
        self._ttft_count: int = 0
        self._tbt_count: int = 0
        self._ttft_max_ms: float = 0.0
        self._tbt_max_ms: float = 0.0

    def update_ttft(self, ttft_ms: float):
        """Update TTFT EMA with burst filtering."""
        if self._ttft_count > 5 and ttft_ms > self._ttft_ema_ms * self._burst_multiplier:
            return  # burst spike, skip
        if self._ttft_count == 0:
            self._ttft_ema_ms = ttft_ms
        else:
            self._ttft_ema_ms = self._alpha * ttft_ms + (1 - self._alpha) * self._ttft_ema_ms
        self._ttft_max_ms = max(self._ttft_max_ms, ttft_ms)
        self._ttft_count += 1

    def update_tbt(self, tbt_ms: float):
        """Update TBT EMA with burst filtering."""
        if self._tbt_count > 5 and tbt_ms > self._tbt_ema_ms * self._burst_multiplier:
            return
        if self._tbt_count == 0:
            self._tbt_ema_ms = tbt_ms
        else:
            self._tbt_ema_ms = self._alpha * tbt_ms + (1 - self._alpha) * self._tbt_ema_ms
        self._tbt_max_ms = max(self._tbt_max_ms, tbt_ms)
        self._tbt_count += 1

    @property
    def predicted_ttft_ms(self) -> float:
        return self._ttft_ema_ms

    @property
    def predicted_tbt_ms(self) -> float:
        return self._tbt_ema_ms

    @property
    def ttft_sample_count(self) -> int:
        return self._ttft_count

    @property
    def tbt_sample_count(self) -> int:
        return self._tbt_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ttft_ema_ms": round(self._ttft_ema_ms, 2),
            "tbt_ema_ms": round(self._tbt_ema_ms, 2),
            "ttft_max_ms": round(self._ttft_max_ms, 2),
            "tbt_max_ms": round(self._tbt_max_ms, 2),
            "ttft_samples": self._ttft_count,
            "tbt_samples": self._tbt_count,
        }


# ── 2. SLOBudgetController ──────────────────────────────────────────────


class BGDecision(enum.Enum):
    """Decision enum for SLOBudgetController."""
    ALLOW = "allow"             # Full BG chunk allowed
    ALLOW_LIMITED = "allow_limited"  # BG allowed with reduced chunk
    DENY = "deny"               # BG not allowed this tick
    SKIP = "skip"               # BG decode should be skipped


class SLOBudgetController:
    """
    Core decision maker. Computes time/chunk/memory budgets and returns a BGDecision.

    Algorithm:
      budget = SLO_target - predicted_FG_p99
      budget > 0 => ALLOW (or ALLOW_LIMITED if FG queue is non-empty)
      budget <= 0 => DENY
    """

    WARMUP_THRESHOLD = 10

    def __init__(self, config: LookaheadConfig):
        self.config = config
        self._warmup_samples: int = 0

    def record_fg_sample(self):
        """Called when a new FG TTFT sample is observed."""
        self._warmup_samples += 1

    def decide(
        self,
        predicted_ttft_ms: float,
        fg_queue_len: int,
        free_kv_blocks: int,
        total_kv_blocks: int,
    ) -> Tuple[BGDecision, float, int, int]:
        """
        Returns (decision, t_budget_ms, c_chunk_tokens, m_free_blocks).
        """
        cfg = self.config

        if self._warmup_samples < self.WARMUP_THRESHOLD:
            return (
                BGDecision.ALLOW_LIMITED,
                cfg.slo_target_ms * 0.2,
                cfg.bg_max_chunk_tokens // 4,
                0,
            )

        # Time budget
        t_budget = cfg.slo_target_ms - predicted_ttft_ms

        # Memory budget
        fg_reserve = int(total_kv_blocks * cfg.mem_reserve_ratio)
        m_free = max(0, free_kv_blocks - fg_reserve)

        # KV-pressure hard deny (replaces the old fg_queue-based deny):
        # under mixed_chunk, queue length doesn't reflect compute pressure,
        # but writing BG KV into a near-full cache will evict FG decode state.
        kv_pressure_limit = cfg.bg_admission_kv_pressure_ratio
        if total_kv_blocks > 0 and kv_pressure_limit > 0:
            kv_used_ratio = 1.0 - (free_kv_blocks / total_kv_blocks)
            if kv_used_ratio >= kv_pressure_limit:
                return BGDecision.DENY, t_budget, 0, m_free

        if t_budget <= 0:
            return BGDecision.DENY, t_budget, 0, m_free

        if m_free <= 0:
            return BGDecision.DENY, t_budget, 0, m_free

        # FG backlog is now a soft signal: large queue → tighter chunk so
        # BG doesn't bloat the mixed forward pass, but we still admit it.
        backlog_thr = cfg.bg_admission_fg_backlog_threshold
        if backlog_thr > 0 and fg_queue_len >= backlog_thr:
            c_chunk = max(1, cfg.bg_max_chunk_tokens // 4)
            return BGDecision.ALLOW_LIMITED, t_budget, c_chunk, m_free

        if fg_queue_len > 0:
            c_chunk = cfg.bg_max_chunk_tokens // 2
            return BGDecision.ALLOW_LIMITED, t_budget, c_chunk, m_free

        return BGDecision.ALLOW, t_budget, cfg.bg_max_chunk_tokens, m_free

    def should_allow_bg_decode(self, predicted_ttft_ms: float, fg_queue_len: int) -> bool:
        """Gate for BG decode: only when no FG pressure and positive time budget."""
        t_budget = self.config.slo_target_ms - predicted_ttft_ms
        return fg_queue_len == 0 and t_budget > 0

    def should_admit_bg(
        self,
        fg_queue_len: int,
        free_kv_blocks: int,
        total_kv_blocks: int,
    ) -> Tuple[bool, str]:
        """Check whether a new BG request should be admitted.

        Under mixed_chunk, BG prefill rides FG decode in one forward pass, so
        queue length alone does not represent contention. The real backstop is
        KV cache pressure: if the running batch already occupies most of the
        cache, admitting BG risks evicting in-flight FG decode state.
        """
        cfg = self.config
        kv_pressure_limit = cfg.bg_admission_kv_pressure_ratio
        if total_kv_blocks > 0 and kv_pressure_limit > 0:
            kv_used_ratio = 1.0 - (free_kv_blocks / total_kv_blocks)
            if kv_used_ratio >= kv_pressure_limit:
                return False, (
                    f"kv_pressure: used={kv_used_ratio:.2f}>={kv_pressure_limit:.2f}"
                )
        fg_reserve = int(total_kv_blocks * cfg.mem_reserve_ratio)
        if free_kv_blocks <= fg_reserve:
            return False, f"mem_pressure: free={free_kv_blocks}<=reserve={fg_reserve}"
        return True, "admitted"


# ── 3. ChunkMixer ────────────────────────────────────────────────────────


class ChunkMixer:
    """
    TBT-guided BG chunk sizing.
    If current TBT exceeds target, reduce chunk size proportionally.
    """

    def __init__(self, config: LookaheadConfig):
        self.tbt_target_ms = config.chunk_mixer_tbt_target_ms
        self.min_ratio = config.chunk_mixer_min_ratio
        self.base_chunk = config.bg_max_chunk_tokens

    def compute_chunk_size(self, current_tbt_ms: float, base_chunk: Optional[int] = None) -> int:
        """
        Compute adjusted chunk size based on TBT pressure.
        Returns chunk tokens in [min_ratio * base, base].
        """
        if base_chunk is None:
            base_chunk = self.base_chunk

        if current_tbt_ms <= 0 or self.tbt_target_ms <= 0:
            return base_chunk

        ratio = self.tbt_target_ms / max(current_tbt_ms, 1e-6)
        ratio = max(self.min_ratio, min(1.0, ratio))
        return max(1, int(base_chunk * ratio))


# ── 4. SoftHardDetector ──────────────────────────────────────────────────


class ReduceSignalLevel(enum.Enum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"


@dataclasses.dataclass
class ReduceSignal:
    level: ReduceSignalLevel
    ctx_ratio: float = 0.0
    session_id: Optional[str] = None


class SoftHardDetector:
    """
    Detects whether a session's context length has crossed soft/hard thresholds.
    Used to trigger ReduceSignal on requests.
    """

    def __init__(self, config: LookaheadConfig):
        self.soft_ratio = config.soft_ctx_ratio
        self.hard_ratio = config.hard_ctx_ratio

    def detect(self, ctx_len: int, capacity: int, session_id: Optional[str] = None) -> ReduceSignal:
        if capacity <= 0:
            return ReduceSignal(level=ReduceSignalLevel.NONE)

        ratio = ctx_len / capacity
        if ratio >= self.hard_ratio:
            return ReduceSignal(
                level=ReduceSignalLevel.HARD,
                ctx_ratio=ratio,
                session_id=session_id,
            )
        elif ratio >= self.soft_ratio:
            return ReduceSignal(
                level=ReduceSignalLevel.SOFT,
                ctx_ratio=ratio,
                session_id=session_id,
            )
        return ReduceSignal(level=ReduceSignalLevel.NONE, ctx_ratio=ratio)


# ── 5. AntiStarvationGuard ───────────────────────────────────────────────


class AntiStarvationGuard:
    """
    Prevents BG requests from starving indefinitely.
    Tracks per-request enqueue time. If the oldest BG request has waited
    longer than max_wait_s, force-override DENY -> ALLOW_LIMITED.
    """

    def __init__(self, config: LookaheadConfig):
        self.max_wait_s = config.anti_starvation_max_wait_s
        self._enqueue_times: Dict[str, float] = {}

    def on_enqueue(self, rid: str):
        self._enqueue_times[rid] = time.monotonic()

    def on_dequeue(self, rid: str):
        self._enqueue_times.pop(rid, None)

    def should_force_allow(self, bg_queue_rids: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Check if any BG request has waited too long.
        Returns (should_force, starving_rid).
        """
        now = time.monotonic()
        for rid in bg_queue_rids:
            enq_time = self._enqueue_times.get(rid)
            if enq_time is not None and (now - enq_time) >= self.max_wait_s:
                return True, rid
        return False, None

    def get_oldest_wait_s(self, bg_queue_rids: List[str]) -> float:
        if not bg_queue_rids:
            return 0.0
        now = time.monotonic()
        waits = [now - self._enqueue_times.get(r, now) for r in bg_queue_rids]
        return max(waits) if waits else 0.0


# ── 6. DriftDetector ─────────────────────────────────────────────────────


class DriftDetector:
    """
    Detects FG latency trend shifts using a sliding window.
    If recent mean > historical mean * threshold_ratio, signals drift.
    """

    def __init__(self, config: LookaheadConfig):
        self.window_size = config.drift_window_size
        self.threshold_ratio = config.drift_threshold_ratio
        self._history: Deque[float] = collections.deque(maxlen=config.drift_window_size * 2)
        self._is_drifting = False

    def update(self, latency_ms: float):
        self._history.append(latency_ms)
        if len(self._history) < self.window_size * 2:
            self._is_drifting = False
            return

        n = self.window_size
        old_window = list(self._history)[:n]
        new_window = list(self._history)[n:]
        old_mean = sum(old_window) / len(old_window) if old_window else 1.0
        new_mean = sum(new_window) / len(new_window) if new_window else 0.0
        self._is_drifting = new_mean > old_mean * self.threshold_ratio

    @property
    def is_drifting(self) -> bool:
        return self._is_drifting

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_drifting": self._is_drifting,
            "history_len": len(self._history),
            "window_size": self.window_size,
        }


# ── 7. ExtremeBurstGuard ─────────────────────────────────────────────────


class ExtremeBurstGuard:
    """
    Extreme burst protection. When a latency spike exceeds
    spike_multiplier * EMA, enters cooldown for cooldown_ticks.
    During cooldown, BG is completely blocked.
    """

    def __init__(self, config: LookaheadConfig):
        self.spike_multiplier = config.burst_spike_multiplier
        self.cooldown_ticks = config.burst_cooldown_ticks
        self._remaining_cooldown: int = 0

    def check_spike(self, current_ms: float, ema_ms: float) -> bool:
        """Check if current latency is a spike. Returns True if in cooldown."""
        if ema_ms > 0 and current_ms > ema_ms * self.spike_multiplier:
            self._remaining_cooldown = self.cooldown_ticks
            return True
        return False

    def tick(self):
        """Called every scheduling tick. Decrements cooldown."""
        if self._remaining_cooldown > 0:
            self._remaining_cooldown -= 1

    @property
    def in_cooldown(self) -> bool:
        return self._remaining_cooldown > 0

    @property
    def remaining_cooldown(self) -> int:
        return self._remaining_cooldown

    def to_dict(self) -> Dict[str, Any]:
        return {
            "in_cooldown": self.in_cooldown,
            "remaining_cooldown": self._remaining_cooldown,
        }


# ── 8. BGSchedulingStats ─────────────────────────────────────────────────


@dataclasses.dataclass
class BGSchedulingStats:
    """Aggregated statistics for BG scheduling. JSON-serializable."""

    # Admission
    bg_admitted_total: int = 0
    bg_denied_total: int = 0

    # Scheduling
    bg_prefill_tokens_total: int = 0
    bg_decode_tokens_total: int = 0
    bg_decode_skipped_total: int = 0
    bg_completed_total: int = 0

    # Anti-starvation
    bg_force_allowed_total: int = 0

    # Burst
    bg_burst_cooldown_triggers: int = 0

    # Queue
    bg_queue_size: int = 0
    bg_chunked_active: bool = False

    # Per-tick token breakdown
    last_tick_seq: int = 0
    last_tick_ts_unix: float = 0.0
    last_tick_gap_ms: float = 0.0
    last_forward_elapsed_ms: float = 0.0
    last_forward_elapsed_source: str = ""
    fg_prefill_tokens_last_tick: int = 0
    bg_prefill_tokens_last_tick: int = 0
    fg_decode_tokens_last_tick: int = 0
    bg_decode_tokens_last_tick: int = 0
    fg_prefill_reqs_last_tick: int = 0
    bg_prefill_reqs_last_tick: int = 0
    fg_prefill_prefix_avg_last_tick: float = 0.0
    bg_prefill_prefix_avg_last_tick: float = 0.0
    fg_prefill_prefix_max_last_tick: int = 0
    bg_prefill_prefix_max_last_tick: int = 0
    tick_history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    tick_history_max_len: int = 256

    # Per-tick BG activity flag (true iff BG extend happened in the most
    # recent tick). Prefer this over bg_prefill_tokens_last_tick, which is
    # latched and does not reset every tick.
    bg_prefill_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def update_queue_state(self, bg_queue_len: int, chunked_active: bool):
        self.bg_queue_size = bg_queue_len
        self.bg_chunked_active = chunked_active


# ── 9. BatchLatencyBuffer ────────────────────────────────────────────────


@dataclasses.dataclass
class BatchLatencySample:
    """One observed (features, latency) pair from a completed forward pass."""
    sum_prefill_len_sq: float
    sum_prefill_len: float
    prefill_bs: int
    sum_decode_len_sq: float
    sum_decode_len: float
    decode_bs: int
    latency_ms: float


class BatchLatencyBuffer:
    """
    Sliding-window buffer for batch latency samples.
    Collects (features, latency) pairs used to fit BatchLatencyModel.
    """

    def __init__(self, max_size: int = 500):
        self._max_size = max_size
        self._samples: Deque[BatchLatencySample] = collections.deque(maxlen=max_size)

    def add(self, sample: BatchLatencySample):
        self._samples.append(sample)

    @property
    def size(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> Deque[BatchLatencySample]:
        return self._samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self._samples),
            "max_size": self._max_size,
        }


# ── 10. BatchLatencyModel ────────────────────────────────────────────────


class BatchLatencyModel:
    """
    HyGen-style quadratic batch latency predictor.

    Model:
      latency_ms = a1*sum_pf_len^2 + b1*sum_pf_len + c1*pf_bs
                 + a2*sum_dc_len^2 + b2*sum_dc_len + c2*dc_bs + d

    Supports two modes:
      1. Pre-calibrated: coefficients injected via LookaheadConfig
      2. Online fitting: periodically fit from BatchLatencyBuffer using least-squares
    """

    def __init__(self, config: LookaheadConfig, buffer: BatchLatencyBuffer):
        self._buffer = buffer
        self._min_samples = config.blp_min_samples
        self._refit_interval = config.blp_refit_interval
        self._samples_at_last_fit = 0

        precal = [config.blp_a1, config.blp_b1, config.blp_c1,
                  config.blp_a2, config.blp_b2, config.blp_c2, config.blp_d]
        has_precal = any(abs(c) > 1e-15 for c in precal)

        if has_precal:
            self._coeffs = precal
            self._fitted = True
            self._mode = "precalibrated"
        else:
            self._coeffs = [0.0] * 7
            self._fitted = False
            self._mode = "online"

    @property
    def is_ready(self) -> bool:
        return self._fitted

    def predict(
        self,
        sum_prefill_len_sq: float,
        sum_prefill_len: float,
        prefill_bs: int,
        sum_decode_len_sq: float,
        sum_decode_len: float,
        decode_bs: int,
    ) -> float:
        """Predict batch latency in ms. Returns 0 if not yet fitted."""
        if not self._fitted:
            return 0.0
        a1, b1, c1, a2, b2, c2, d = self._coeffs
        return (a1 * sum_prefill_len_sq + b1 * sum_prefill_len + c1 * prefill_bs
                + a2 * sum_decode_len_sq + b2 * sum_decode_len + c2 * decode_bs + d)

    def predict_seq(self, prefill_len: Optional[int] = None,
                    decode_len: Optional[int] = None) -> float:
        """Predict latency increment for a single sequence (prefill or decode)."""
        if not self._fitted:
            return 0.0
        a1, b1, c1, a2, b2, c2, _d = self._coeffs
        if prefill_len is not None:
            return a1 * prefill_len * prefill_len + b1 * prefill_len + c1
        if decode_len is not None:
            return a2 * decode_len * decode_len + b2 * decode_len + c2
        return 0.0

    def maybe_refit(self):
        """Refit from buffer if enough new samples have accumulated."""
        if self._mode == "precalibrated":
            return
        n = self._buffer.size
        if n < self._min_samples:
            return
        if n - self._samples_at_last_fit < self._refit_interval:
            return
        self._fit_from_buffer()

    def _fit_from_buffer(self):
        """Least-squares fit of the 7-coefficient model from buffered samples."""
        try:
            import numpy as np
        except ImportError:
            return

        samples = list(self._buffer.samples)
        n = len(samples)
        if n < 7:
            return

        X = np.zeros((n, 7), dtype=np.float64)
        y = np.zeros(n, dtype=np.float64)
        for i, s in enumerate(samples):
            X[i, 0] = s.sum_prefill_len_sq
            X[i, 1] = s.sum_prefill_len
            X[i, 2] = s.prefill_bs
            X[i, 3] = s.sum_decode_len_sq
            X[i, 4] = s.sum_decode_len
            X[i, 5] = s.decode_bs
            X[i, 6] = 1.0
            y[i] = s.latency_ms

        result, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        self._coeffs = result.tolist()
        self._fitted = True
        self._samples_at_last_fit = n

    def get_max_prefill_tokens(self, latency_budget_ms: float) -> int:
        """
        Solve a1*x^2 + b1*x + c1 <= latency_budget for maximum x.
        Useful for determining how many BG prefill tokens fit in the budget.
        """
        if not self._fitted:
            return 0
        a1, b1, c1, *_ = self._coeffs
        budget = latency_budget_ms - c1
        if a1 <= 1e-15:
            if b1 > 1e-15:
                return max(0, int(budget / b1))
            return 0
        disc = b1 * b1 + 4.0 * a1 * budget
        if disc < 0:
            return 0
        x = (-b1 + math.sqrt(disc)) / (2.0 * a1)
        return max(0, int(x))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self._mode,
            "fitted": self._fitted,
            "coeffs": {
                "a1": self._coeffs[0], "b1": self._coeffs[1], "c1": self._coeffs[2],
                "a2": self._coeffs[3], "b2": self._coeffs[4], "c2": self._coeffs[5],
                "d": self._coeffs[6],
            },
            "samples_at_last_fit": self._samples_at_last_fit,
        }
