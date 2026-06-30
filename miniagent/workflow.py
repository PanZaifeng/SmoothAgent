"""MiniAgent framework workflow row helpers."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any

from miniagent.core import (
    Message,
    StrategyConfig,
    StrategyName,
    assistant_message,
    approximate_token_count,
    make_initial_messages,
    serialize_messages,
    should_lookahead,
    step_user_message,
    tool_message,
    transform_messages,
)
from miniagent.sglang import ChatResult, SmoothAgentRequestMeta

CSV_FIELDS = [
    "n_agents",
    "strategy",
    "mode",
    "agent_id",
    "step",
    "ttft_ms",
    "first_byte_ms",
    "prepare_ms",
    "prefill_ms",
    "ctx_tokens",
    "post_tokens",
    "is_commit",
    "bg_status",
    "tbt_avg_ms",
    "tbt_p99_ms",
    "n_decode_tokens",
]


@dataclass(slots=True)
class MiniAgentState:
    messages: list[Message]
    completed_commits: int = 0
    latest_task_id: str = ""
    last_summary_text: str = ""


@dataclass(slots=True)
class MiniAgentStep:
    row: dict[str, Any]
    raw: dict[str, Any]


def initial_state(agent_id: int, seed: int) -> MiniAgentState:
    return MiniAgentState(messages=make_initial_messages(agent_id, seed))


def percentile(values: list[float], p: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v) and v >= 0)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    k = (len(clean) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return clean[lo]
    return clean[lo] + (clean[hi] - clean[lo]) * (k - lo)


def request_meta(
    *,
    run_id: str,
    strategy: StrategyName,
    mode: str,
    agent_id: int,
    step: int,
    background: bool,
) -> SmoothAgentRequestMeta:
    return SmoothAgentRequestMeta(
        request_class="bg" if background else "fg",
        lookahead_group_id=f"{run_id}:miniagent:{strategy}:{mode}:a{agent_id}:s{step}",
    )


def build_step_row(
    *,
    run_id: str,
    n_agents: int,
    strategy: StrategyName,
    mode: str,
    agent_id: int,
    step: int,
    prompt: str,
    is_commit: bool,
    prepare_ms: float,
    result: ChatResult,
    bg_status: str,
    tbt_ms: list[float] | None = None,
) -> MiniAgentStep:
    token_count = int(result.meta.get("prompt_tokens") or approximate_token_count(prompt))
    completion_tokens = int(result.meta.get("completion_tokens") or 0)
    tbt = tbt_ms or []
    tbt_avg = sum(tbt) / len(tbt) if tbt else 0.0
    ttft_ms = prepare_ms + result.ttft_ms if result.ttft_ms >= 0 else -1.0
    row = {
        "n_agents": n_agents,
        "strategy": strategy,
        "mode": mode,
        "agent_id": agent_id,
        "step": step,
        "ttft_ms": f"{ttft_ms:.1f}",
        "first_byte_ms": f"{ttft_ms:.1f}",
        "prepare_ms": f"{prepare_ms:.1f}",
        "prefill_ms": f"{result.ttft_ms:.1f}",
        "ctx_tokens": token_count,
        "post_tokens": token_count,
        "is_commit": str(bool(is_commit)),
        "bg_status": bg_status,
        "tbt_avg_ms": f"{tbt_avg:.1f}",
        "tbt_p99_ms": f"{percentile(tbt, 99):.1f}",
        "n_decode_tokens": completion_tokens,
    }
    raw = {
        "kind": "step",
        "run_id": run_id,
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "n_agents": n_agents,
        "strategy": strategy,
        "mode": mode,
        "agent_id": agent_id,
        "step": step,
        "is_commit": is_commit,
        "ttft_ms": ttft_ms,
        "prepare_ms": prepare_ms,
        "prefill_ms": result.ttft_ms,
        "duration_ms": result.duration_ms,
        "prompt_tokens": token_count,
        "completion_tokens": completion_tokens,
        "server_rid": result.server_rid,
        "server_meta": result.meta,
        "bg_status": bg_status,
    }
    return MiniAgentStep(row=row, raw=raw)


def sync_commit_projection(
    state: MiniAgentState,
    strategy: StrategyName,
    cfg: StrategyConfig,
    *,
    summary_text: str = "",
) -> tuple[str, dict[str, Any]]:
    transformed = transform_messages(strategy, state.messages, cfg, summary_text)
    state.messages = transformed.messages
    state.completed_commits += 1
    return serialize_messages(transformed.messages), transformed.metadata


def append_workflow_turn(state: MiniAgentState, cfg: StrategyConfig, step: int) -> None:
    state.messages.append(step_user_message(cfg.agent_id, step, cfg.seed))
    state.messages.append(assistant_message(cfg.agent_id, step, cfg))
    state.messages.append(tool_message(cfg.agent_id, step, cfg))


def should_schedule_background(
    *,
    state: MiniAgentState,
    step: int,
    next_commit_step: int | None,
    strategy: StrategyName,
    warmup_lead_steps: dict[str, int],
    summarize_soft_trigger_tokens: int = 0,
) -> bool:
    return should_lookahead(
        step=step,
        next_commit_step=next_commit_step,
        strategy=strategy,
        warmup_lead_steps=warmup_lead_steps,
        messages=state.messages,
        summarize_soft_trigger_tokens=summarize_soft_trigger_tokens,
    )
