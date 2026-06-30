"""State containers for lookahead memory transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from llama_index.core.base.llms.types import ChatMessage


@dataclass
class MainState:
    """Working context observed by the foreground agent stream."""

    messages: list[ChatMessage] = field(default_factory=list)
    token_count: int = 0
    turn_count: int = 0
    strategy_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class LookaheadArtifact:
    """Completed transform output tied to the source message signature."""

    transformed: list[ChatMessage]
    source_signature: str
    task_id: str
    generation: int


@dataclass
class LookaheadState:
    """Background transform state maintained by the lookahead runtime."""

    transformed: list[ChatMessage] = field(default_factory=list)
    last_segment_end: int = 0
    pending_task_id: Optional[str] = None
    scheduled_signature: Optional[str] = None
    completed_signature: Optional[str] = None
    completed: bool = False
    artifact: Optional[LookaheadArtifact] = None
    error: Optional[BaseException] = None
    generation: int = 0
    strategy_data: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear pending/completed lookahead state after commit or reset."""
        self.transformed = []
        self.last_segment_end = 0
        self.pending_task_id = None
        self.scheduled_signature = None
        self.completed_signature = None
        self.completed = False
        self.artifact = None
        self.error = None
        self.strategy_data = {}
