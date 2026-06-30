"""Build :class:`BatchSnapshot` from scheduler-state duck types.

These helpers are extracted from
:class:`sglang.srt.managers.scheduler_lookahead_mixin.SchedulerLookaheadMixin`
so they can be exercised by unit tests without dragging the full SGLang
runtime (orjson/torch/etc.). The mixin holds onto SGLang-specific
references (running batch, waiting queue, BG queue) and forwards the
relevant pieces here.

The snapshot is the input to the ``eq:latency-model`` batch-latency
estimator and to :class:`SmoothAgentAdmissionController` admission decisions.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional

from sglang.srt.managers.smoothagent.admission_controller import BatchSnapshot
from sglang.srt.managers.smoothagent.batch_latency_estimator import (
    DecodeRequest,
    PrefillChunk,
)


# ---------------------------------------------------------------------------
# Per-request estimators
# ---------------------------------------------------------------------------


def paper_remaining_prefill_tokens(req: Any) -> int:
    """Best-effort count of tokens still pending prefill for ``req``.

    A request's ``origin_input_ids`` is the full prompt; ``fill_ids`` and
    ``kv_committed_len`` reflect chunked-prefill progress so far.
    """
    origin = getattr(req, "origin_input_ids", None)
    total = len(origin) if origin is not None else 0
    already_filled = max(
        int(getattr(req, "kv_committed_len", 0) or 0),
        len(getattr(req, "fill_ids", []) or []),
    )
    return max(0, total - already_filled)


def paper_estimate_prefix_kv_len(req: Any) -> int:
    """KV prefix length already cached when this request's chunk fires.

    Uses ``kv_committed_len`` (true KV occupancy) when set; falls back to
    the radix prefix indices length when the request has only matched the
    cache but not committed yet.
    """
    kv = int(getattr(req, "kv_committed_len", 0) or 0)
    if kv > 0:
        return kv
    prefix = getattr(req, "prefix_indices", None)
    if prefix is None:
        return 0
    try:
        return int(prefix.shape[0])
    except Exception:
        try:
            return int(len(prefix))
        except Exception:
            return 0


def paper_estimate_next_chunk_tokens(
    req: Any,
    *,
    bg_max_chunk_tokens: int,
    predicted_tbt_ms: float = 0.0,
    chunk_mixer_compute: Optional[Callable[[float, int], int]] = None,
) -> int:
    """Estimate the chunk size that ``maybe_append_bg_prefill`` would issue.

    The base size is ``bg_max_chunk_tokens``, optionally shaped down by
    :class:`ChunkMixer` when the latest TBT measurement is high. The
    result is then capped by the request's actual unprefilled tokens.
    """
    base = int(bg_max_chunk_tokens)
    if predicted_tbt_ms > 0.0 and chunk_mixer_compute is not None:
        try:
            base = int(chunk_mixer_compute(predicted_tbt_ms, base))
        except Exception:
            pass
    base = max(1, base)
    unprefilled = paper_remaining_prefill_tokens(req)
    if unprefilled > 0:
        return max(1, min(base, unprefilled))
    ctx_hint = int(getattr(req, "ctx_len_tokens", 0) or 0)
    return max(1, min(base, ctx_hint or base))


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------


def build_paper_batch_snapshot(
    *,
    running_reqs: Iterable[Any],
    waiting_queue: Iterable[Any],
    bg_chunked_req: Optional[Any] = None,
    bg_chunked_reqs: Optional[Iterable[Any]] = None,
    bg_waiting_queue: Iterable[Any],
    bg_max_chunk_tokens: int,
    predicted_tbt_ms: float = 0.0,
    chunk_mixer_compute: Optional[Callable[[float, int], int]] = None,
) -> BatchSnapshot:
    """Build a :class:`BatchSnapshot` reflecting the live mixed-chunk state.

    - ``decodes``: pure-decode reqs from ``running_reqs`` (those with
      ``extend_input_len == 0``). ``kv_cache_len`` counts both committed
      input KV and produced output tokens.
    - ``lc_chunks``: in-flight FG prefill chunks plus FG queued reqs
      (modeled as their next chunk).
    - ``be_chunks``: every BG chunked req in flight (paper-mode multi)
      plus ``bg_waiting_queue`` reqs (modeled as their next chunk).
    - ``waiting_queue_len``: length of the FG waiting queue.

    The legacy ``bg_chunked_req`` (singular) is kept for callers that
    haven't migrated to the list form. If both are provided,
    ``bg_chunked_reqs`` wins. Pass ``None`` to ``bg_chunked_req`` and
    the list to ``bg_chunked_reqs`` for paper-mode multi-chunked.
    """
    waiting_list: List[Any] = list(waiting_queue)
    bg_waiting_list: List[Any] = list(bg_waiting_queue)

    decodes: List[DecodeRequest] = []
    lc_chunks: List[PrefillChunk] = []

    for r in running_reqs:
        ext = int(getattr(r, "extend_input_len", 0) or 0)
        if ext > 0:
            lc_chunks.append(
                PrefillChunk(
                    chunk_tokens=max(1, ext),
                    prefix_kv_len=max(0, paper_estimate_prefix_kv_len(r)),
                )
            )
            continue
        kv_committed = int(getattr(r, "kv_committed_len", 0) or 0)
        output_len = len(getattr(r, "output_ids", []) or [])
        kv = kv_committed + output_len
        if kv > 0:
            decodes.append(DecodeRequest(kv_cache_len=kv))

    for r in waiting_list:
        chunk_tokens = paper_estimate_next_chunk_tokens(
            r,
            bg_max_chunk_tokens=bg_max_chunk_tokens,
            predicted_tbt_ms=predicted_tbt_ms,
            chunk_mixer_compute=chunk_mixer_compute,
        )
        lc_chunks.append(
            PrefillChunk(
                chunk_tokens=max(1, chunk_tokens),
                prefix_kv_len=max(0, paper_estimate_prefix_kv_len(r)),
            )
        )

    be_chunks: List[PrefillChunk] = []
    if bg_chunked_reqs is not None:
        bg_chunked_iterable = list(bg_chunked_reqs)
    elif bg_chunked_req is not None:
        bg_chunked_iterable = [bg_chunked_req]
    else:
        bg_chunked_iterable = []
    for r in bg_chunked_iterable:
        ext = int(getattr(r, "extend_input_len", 0) or 0)
        chunk_tokens = (
            ext
            if ext > 0
            else paper_estimate_next_chunk_tokens(
                r,
                bg_max_chunk_tokens=bg_max_chunk_tokens,
                predicted_tbt_ms=predicted_tbt_ms,
                chunk_mixer_compute=chunk_mixer_compute,
            )
        )
        be_chunks.append(
            PrefillChunk(
                chunk_tokens=max(1, chunk_tokens),
                prefix_kv_len=max(0, paper_estimate_prefix_kv_len(r)),
            )
        )

    for r in bg_waiting_list:
        be_chunks.append(
            PrefillChunk(
                chunk_tokens=max(
                    1,
                    paper_estimate_next_chunk_tokens(
                        r,
                        bg_max_chunk_tokens=bg_max_chunk_tokens,
                        predicted_tbt_ms=predicted_tbt_ms,
                        chunk_mixer_compute=chunk_mixer_compute,
                    ),
                ),
                prefix_kv_len=max(0, paper_estimate_prefix_kv_len(r)),
            )
        )

    return BatchSnapshot(
        decodes=decodes,
        lc_chunks=lc_chunks,
        be_chunks=be_chunks,
        waiting_queue_len=len(waiting_list),
    )
