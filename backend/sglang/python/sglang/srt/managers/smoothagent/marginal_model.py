"""Marginal-cost models for the per-LC slack formula.

The slack-aware admission gate (``scheduler_lookahead_mixin._per_lc_slack_ms``)
charges every in-flight BE chunk a *marginal* cost — the extra wall-time it
imposes on overlapping LC chunks. v6 used a flat constant; this module
exposes the calibrated alternatives:

- :func:`closed_form_marginal_ms` — linear-attention fit
  ``c0 + c_attn · (q · prefix + q·(q+1)/2)`` from
  ``runs/calibration_v2/marginal_pairs.jsonl`` (N_lc ≥ 2 fit, 2026-05-12).
- :func:`eq1_marginal_ms` — marginal shaped like ``eq:latency-model`` using
  a separate ``T_GEMM`` lookup callable + ``α_p`` (kept here as a thin
  convenience for callers that already hold an estimator).

Both functions are pure: no scheduler state, no env reads. The mixin is
responsible for reading env / JSON config and passing concrete numbers in.
"""
from __future__ import annotations

from typing import Callable


def closed_form_marginal_ms(
    *,
    q_be: int,
    prefix_be: int,
    c0: float = 1.25,
    c_attn: float = 1.93e-6,
) -> float:
    """Predicted per-tick marginal cost (ms) of one BE chunk.

    ``q_be``  — tokens in the BE chunk this tick (clamped to ``chunk_cap``).
    ``prefix_be`` — KV tokens already committed for this BE before the chunk.
    ``c0`` — intercept covering per-chunk kernel-launch + sync overhead.
    ``c_attn`` — coefficient on prefill attention work ``q·prefix + q²/2``.

    Defaults are the N_lc≥2 linear_attn fit (R²=0.22, MAE=2.29 ms).
    """
    q = max(0, int(q_be))
    prefix = max(0, int(prefix_be))
    if q == 0:
        return 0.0
    attn = q * prefix + q * (q + 1) // 2
    return max(0.0, c0 + c_attn * attn)


def eq1_marginal_ms(
    *,
    q_be: int,
    prefix_be: int,
    m_base: int,
    alpha_p: float,
    gemm_cost: Callable[[int], float],
    n_other: int = 0,
    gamma_launch_ms: float = 0.0,
    kappa: float = 0.0,
) -> float:
    """Marginal from ``eq:latency-model``, optionally with concurrency scaling.

    ``T_GEMM(M+q) − T_GEMM(M) + α_p·A_BE`` is recovered when
    ``n_other=0``, ``gamma_launch_ms=0``, and ``kappa=0``. Set
    ``gamma_launch_ms > 0`` and ``kappa > 0`` to enable the hybrid
    concurrency-aware extension:

    ``γ_launch + Δ_GEMM + α_p·A_BE·(1 + κ·N_other)``.

    ``gemm_cost`` is typically ``BatchLatencyEstimator.gemm_cost``.
    """
    q = max(0, int(q_be))
    prefix = max(0, int(prefix_be))
    if q == 0:
        return 0.0
    attn = q * prefix + q * (q + 1) // 2
    gemm_delta = float(gemm_cost(m_base + q)) - float(gemm_cost(m_base))
    scale = 1.0 + max(0.0, float(kappa)) * max(0, int(n_other))
    return max(0.0, float(gamma_launch_ms) + gemm_delta + alpha_p * attn * scale)


__all__ = [
    "closed_form_marginal_ms",
    "eq1_marginal_ms",
]
