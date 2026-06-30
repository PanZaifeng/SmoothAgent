"""State containers for the SmoothAgent lookahead runtime.

These follow the abstractions described by the SmoothAgent runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MainState:
    """Working context held by the main agent stream.

    Attributes:
        messages: Live message list used by the agent loop.
        token_count: Approximate token usage of `messages`.
        turn_count: Number of conversational turns observed so far.
        strategy_data: Strategy-private mutable state (e.g., counters).
    """

    messages: list[Any] = field(default_factory=list)
    token_count: int = 0
    turn_count: int = 0
    strategy_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class LookaheadState:
    """Asynchronous transformation state maintained by the runtime.

    Attributes:
        transformed: Transformed messages prepared by the lookahead stream.
        last_segment_end: Index in `MainState.messages` up to which the
            lookahead has already incorporated content.
        pending_task_id: Identifier of the in-flight transform task, or
            ``None`` if no task is running.
        scheduled_signature: Stable signature of the message snapshot used
            to start the in-flight transform.
        completed_signature: Stable signature of the prefix represented by
            ``transformed``.
        completed: Whether `transformed` is ready to be committed.
        error: The last transform exception, if any.
        strategy_data: Strategy-private state (file paths, summary text,
            window boundaries, ...).
    """

    transformed: list[Any] = field(default_factory=list)
    last_segment_end: int = 0
    pending_task_id: str | None = None
    scheduled_signature: str | None = None
    completed_signature: str | None = None
    completed: bool = False
    error: BaseException | None = None
    strategy_data: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        """Reset the lookahead state after a successful commit."""
        self.transformed = []
        self.last_segment_end = 0
        self.pending_task_id = None
        self.scheduled_signature = None
        self.completed_signature = None
        self.completed = False
        self.error = None
        self.strategy_data = {}
