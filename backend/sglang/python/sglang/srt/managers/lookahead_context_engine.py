from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Tuple


@dataclasses.dataclass
class LookaheadMainState:
    ctx_len_tokens: int
    capacity: int
    soft_threshold: int = 0
    hard_threshold: int = 0
    ready: bool = False


@dataclasses.dataclass
class LookaheadState:
    version: int = 0
    pending: bool = False
    ready: bool = False
    task_id: str | None = None
    strategy: str = "sliding_window"
    control: Dict[str, Any] = dataclasses.field(default_factory=dict)


def resolve_soft_threshold(main_state: LookaheadMainState) -> int:
    if main_state.soft_threshold > 0:
        return main_state.soft_threshold
    return max(1, int(main_state.capacity * 0.6))


def resolve_hard_threshold(main_state: LookaheadMainState) -> int:
    if main_state.hard_threshold > 0:
        return main_state.hard_threshold
    return max(1, int(main_state.capacity * 0.8))


def should_trigger(main_state: LookaheadMainState, la_state: LookaheadState) -> bool:
    return (
        main_state.ctx_len_tokens >= resolve_soft_threshold(main_state)
        and not la_state.pending
    )


def transform(
    prefix: Tuple[List[int], Dict[str, Any]],
    la_state: LookaheadState,
) -> tuple[List[int], Dict[str, Any]]:
    input_ids, control = prefix
    artifact = dict(control)
    artifact.setdefault("task_id", la_state.task_id)
    artifact.setdefault("strategy", la_state.strategy)
    artifact.setdefault("drop_count", int(control.get("drop_count", 0)))
    return list(input_ids), artifact


def should_commit(main_state: LookaheadMainState) -> bool:
    return main_state.ctx_len_tokens >= resolve_hard_threshold(main_state)
