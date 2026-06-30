"""
Lookahead BG Scheduling — Configuration Center.

All BG scheduling parameters in one place.
Each parameter can be overridden via SGLANG_LOOKAHEAD_* environment variables.
"""

import dataclasses
import os
from typing import Optional


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(f"SGLANG_LOOKAHEAD_{key}")
    return float(val) if val is not None else default


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(f"SGLANG_LOOKAHEAD_{key}")
    return int(val) if val is not None else default


@dataclasses.dataclass
class LookaheadConfig:
    """
    Single-point configuration for all BG scheduling parameters.

    Design (HyGen + Sarathi-Serve):
      - Sarathi-Serve layer: execution semantics (chunk size, iteration token
        budget). bg_max_chunk_tokens is the per-iteration cap for BG prefill.
      - HyGen layer: co-location control (SLO, latency predictor, two-phase
        schedule FG-first then BG, preemption when FG has pressure).

    Control knob alignment (critical):
      Latency predictor / BatchLatencyModel coefficients must be profiled with
      the same chunk size (bg_max_chunk_tokens), model, and GPU as production.
      If you change chunk size or hardware, re-profile and refit; otherwise
      the SLO budget will be wrong (either waste throughput or violate SLO).
    """

    # --- SLO / Latency ---
    slo_target_ms: float = dataclasses.field(
        default_factory=lambda: _env_float("SLO_TARGET_MS", 500.0)
    )

    # --- BG Chunk / Prefill ---
    # Sarathi-Serve style: per-iteration token cap. Must match what latency
    # predictor / BatchLatencyModel was profiled with (see LookaheadConfig docstring).
    bg_max_chunk_tokens: int = dataclasses.field(
        default_factory=lambda: _env_int("BG_MAX_CHUNK_TOKENS", 512)
    )

    # --- BG Decode ---
    bg_max_decode_tokens_per_tick: int = dataclasses.field(
        default_factory=lambda: _env_int("BG_MAX_DECODE_TOKENS_PER_TICK", 32)
    )
    # Number of unfinished BG prefill requests allowed to hold scheduler req
    # slots at the same time. Without this cap, high-concurrency colocated
    # runs can let BG chunked work occupy all req slots before the foreground
    # commit burst arrives.
    bg_max_inflight_reqs: int = dataclasses.field(
        default_factory=lambda: _env_int("BG_MAX_INFLIGHT_REQS", 2)
    )

    # --- BG-during-FG-decode interleave ---
    # When FG is mid-decode and there is BG prefill work pending, the scheduler
    # steals every Nth tick for a BG-only prefill (`allow_bg_prefill_only`).
    # N=2 → 1 BG prefill tick per 2 FG decode ticks, i.e. FG decode runs at
    # ~67% of baseline TPS while BG catches up. N=0 disables the mechanism
    # (old behavior: BG only runs when system is fully idle).
    bg_prefill_decode_interleave: int = dataclasses.field(
        default_factory=lambda: _env_int("BG_PREFILL_DECODE_INTERLEAVE", 2)
    )

    # --- Admission ---
    # Soft signal under mixed_chunk: when FG waiting_queue grows past this,
    # BG is downgraded (smaller chunk) but NOT denied. With mixed_chunk on,
    # BG prefill rides the FG decode forward and doesn't really compete for
    # compute, so queue length alone is a poor pressure proxy.
    bg_admission_fg_backlog_threshold: int = dataclasses.field(
        default_factory=lambda: _env_int("BG_ADMISSION_FG_BACKLOG_THRESHOLD", 8)
    )
    # Hard backstop: deny BG when running batch's KV usage exceeds this ratio.
    # Replaces the old fg_queue-based hard deny — under mixed_chunk the real
    # pressure is "is the cache about to overflow", not "is the queue long".
    # 0.85 = allow BG until 85% of KV pages are in use.
    bg_admission_kv_pressure_ratio: float = dataclasses.field(
        default_factory=lambda: _env_float("BG_ADMISSION_KV_PRESSURE_RATIO", 0.85)
    )

    # --- ReduceSignal thresholds ---
    soft_ctx_ratio: float = dataclasses.field(
        default_factory=lambda: _env_float("SOFT_CTX_RATIO", 0.6)
    )
    hard_ctx_ratio: float = dataclasses.field(
        default_factory=lambda: _env_float("HARD_CTX_RATIO", 0.8)
    )

    # --- Anti-starvation ---
    anti_starvation_max_wait_s: float = dataclasses.field(
        default_factory=lambda: _env_float("ANTI_STARVATION_MAX_WAIT_S", 30.0)
    )

    # --- EMA ---
    ema_alpha: float = dataclasses.field(
        default_factory=lambda: _env_float("EMA_ALPHA", 0.1)
    )

    # --- Memory reserve ---
    mem_reserve_ratio: float = dataclasses.field(
        default_factory=lambda: _env_float("MEM_RESERVE_RATIO", 0.1)
    )

    # --- Drift detection ---
    drift_window_size: int = dataclasses.field(
        default_factory=lambda: _env_int("DRIFT_WINDOW_SIZE", 20)
    )
    drift_threshold_ratio: float = dataclasses.field(
        default_factory=lambda: _env_float("DRIFT_THRESHOLD_RATIO", 1.5)
    )

    # --- Extreme burst ---
    burst_spike_multiplier: float = dataclasses.field(
        default_factory=lambda: _env_float("BURST_SPIKE_MULTIPLIER", 3.0)
    )
    burst_cooldown_ticks: int = dataclasses.field(
        default_factory=lambda: _env_int("BURST_COOLDOWN_TICKS", 10)
    )

    # --- ChunkMixer TBT guidance ---
    chunk_mixer_tbt_target_ms: float = dataclasses.field(
        default_factory=lambda: _env_float("CHUNK_MIXER_TBT_TARGET_MS", 50.0)
    )
    chunk_mixer_min_ratio: float = dataclasses.field(
        default_factory=lambda: _env_float("CHUNK_MIXER_MIN_RATIO", 0.1)
    )

    # --- Batch latency predictor coefficients (HyGen-style) ---
    # latency_ms = a1*sum_pf_len^2 + b1*sum_pf_len + c1*pf_bs
    #            + a2*sum_dc_len^2 + b2*sum_dc_len + c2*dc_bs + d
    # Set to 0.0 (default) to use online fitting instead of pre-calibrated values.
    blp_a1: float = dataclasses.field(
        default_factory=lambda: _env_float("BLP_A1", 0.0)
    )
    blp_b1: float = dataclasses.field(
        default_factory=lambda: _env_float("BLP_B1", 0.0)
    )
    blp_c1: float = dataclasses.field(
        default_factory=lambda: _env_float("BLP_C1", 0.0)
    )
    blp_a2: float = dataclasses.field(
        default_factory=lambda: _env_float("BLP_A2", 0.0)
    )
    blp_b2: float = dataclasses.field(
        default_factory=lambda: _env_float("BLP_B2", 0.0)
    )
    blp_c2: float = dataclasses.field(
        default_factory=lambda: _env_float("BLP_C2", 0.0)
    )
    blp_d: float = dataclasses.field(
        default_factory=lambda: _env_float("BLP_D", 0.0)
    )
    blp_min_samples: int = dataclasses.field(
        default_factory=lambda: _env_int("BLP_MIN_SAMPLES", 50)
    )
    blp_refit_interval: int = dataclasses.field(
        default_factory=lambda: _env_int("BLP_REFIT_INTERVAL", 50)
    )

    @classmethod
    def from_server_args(cls, server_args) -> "LookaheadConfig":
        """Create config, using server_args overrides where applicable."""
        cfg = cls()
        if hasattr(server_args, "bg_max_chunk_tokens"):
            cfg.bg_max_chunk_tokens = server_args.bg_max_chunk_tokens
        if hasattr(server_args, "bg_admission_fg_backlog_threshold"):
            cfg.bg_admission_fg_backlog_threshold = server_args.bg_admission_fg_backlog_threshold
        return cfg
