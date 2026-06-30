"""SmoothAgent (lookahead context engineering) helpers.

This sub-package adds modules that mirror the scheduler model in the
SmoothAgent paper:

- :mod:`smoothagent_meta`        — :class:`LookaheadRequestMeta` data class
                                   and ``extra_body`` parsing helpers used
                                   by the OpenAI / generate API entry points.
- :mod:`batch_latency_estimator` — Reference implementation of
                                   ``eq:latency-model``
                                   ``T_GEMM(M) + α_d·ΣL + α_p·ΣA_j``.
- :mod:`scheduler_algorithms`    — Reference implementations of
                                   ``alg:schedule-prefill`` (PD
                                   disaggregated) and
                                   ``alg:schedule-hybrid`` (PD co-located)
                                   as side-effect-free
                                   helpers, suitable for unit testing
                                   and as a fallback path for new code.

The pre-existing BG / lookahead controllers in
``sglang/srt/managers/lookahead_controller.py`` and
``scheduler_lookahead_mixin.py`` continue to drive the production
scheduler. The modules here are additive and do not alter the existing
control plane or scheduling path.
"""

from sglang.srt.managers.smoothagent.batch_latency_estimator import (
    BatchLatencyEstimator,
    BatchLatencyEstimatorConfig,
    DecodeRequest,
    PrefillChunk,
    default_gemm_table,
    default_gemm_table_qwen3_8b_h100,
)
from sglang.srt.managers.smoothagent.smoothagent_meta import (
    LOOKAHEAD_META_FIELDS,
    LookaheadRequestMeta,
    parse_extra_body,
)
from sglang.srt.managers.smoothagent.admission_controller import (
    PAPER_MODE_ENV_VAR,
    AdmissionVerdict,
    BatchSnapshot,
    SmoothAgentAdmissionController,
    paper_mode_enabled_default,
)
from sglang.srt.managers.smoothagent.req_field_pipeline import (
    attach_smoothagent_fields,
    req_meta,
)
from sglang.srt.managers.smoothagent.scheduler_algorithms import (
    AdmissionDecision,
    LCRequest,
    schedule_hybrid_colocated,
    schedule_prefill_disaggregated,
)
from sglang.srt.managers.smoothagent.snapshot_builder import (
    build_paper_batch_snapshot,
    paper_estimate_next_chunk_tokens,
    paper_estimate_prefix_kv_len,
    paper_remaining_prefill_tokens,
)

__all__ = [
    "LOOKAHEAD_META_FIELDS",
    "PAPER_MODE_ENV_VAR",
    "AdmissionDecision",
    "AdmissionVerdict",
    "BatchLatencyEstimator",
    "BatchLatencyEstimatorConfig",
    "BatchSnapshot",
    "SmoothAgentAdmissionController",
    "DecodeRequest",
    "LCRequest",
    "LookaheadRequestMeta",
    "PrefillChunk",
    "attach_smoothagent_fields",
    "build_paper_batch_snapshot",
    "default_gemm_table",
    "default_gemm_table_qwen3_8b_h100",
    "paper_estimate_next_chunk_tokens",
    "paper_estimate_prefix_kv_len",
    "paper_mode_enabled_default",
    "paper_remaining_prefill_tokens",
    "parse_extra_body",
    "req_meta",
    "schedule_hybrid_colocated",
    "schedule_prefill_disaggregated",
]
