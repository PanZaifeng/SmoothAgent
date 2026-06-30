import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..models import LLMMessage

PayloadT = TypeVar("PayloadT")
MessageSignature = tuple[str, ...]


@dataclass(frozen=True)
class LookaheadArtifact(Generic[PayloadT]):
    signature: MessageSignature
    payload: PayloadT


@dataclass(frozen=True)
class LookaheadMainState:
    usage: int
    soft_limit: int
    hard_limit: int
    ready: bool
    signature: MessageSignature


@dataclass
class LookaheadState(Generic[PayloadT]):
    task: asyncio.Task[LookaheadArtifact[PayloadT]] | None = None
    scheduled_signature: MessageSignature | None = None
    artifact: LookaheadArtifact[PayloadT] | None = None


class AsyncLookaheadEngine(Generic[PayloadT]):
    def schedule(
        self,
        builder: Callable[[], Awaitable[LookaheadArtifact[PayloadT]]],
        state: LookaheadState[PayloadT],
        signature: MessageSignature,
    ) -> None:
        self.clear(state)
        state.scheduled_signature = signature
        state.task = asyncio.create_task(builder())

    def poll(self, state: LookaheadState[PayloadT]) -> None:
        task = state.task
        if task is None or not task.done():
            return

        state.task = None
        scheduled_signature = state.scheduled_signature
        state.scheduled_signature = None

        try:
            artifact = task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            return

        if scheduled_signature is None or artifact.signature != scheduled_signature:
            return

        state.artifact = artifact

    def consume(self, state: LookaheadState[PayloadT]) -> LookaheadArtifact[PayloadT] | None:
        self.poll(state)
        artifact = state.artifact
        state.artifact = None
        return artifact

    def clear(self, state: LookaheadState[PayloadT]) -> None:
        task = state.task
        if task is not None:
            if task.done():
                with suppress(asyncio.CancelledError, Exception):
                    task.result()
            else:
                task.cancel()
        state.task = None
        state.scheduled_signature = None
        state.artifact = None


def message_signature(messages: Sequence[LLMMessage]) -> MessageSignature:
    return tuple(
        json.dumps(message.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) for message in messages
    )


def should_trigger(main_state: LookaheadMainState, lookahead_state: LookaheadState[PayloadT]) -> bool:
    if main_state.ready:
        return False
    if main_state.usage < main_state.soft_limit and main_state.usage < main_state.hard_limit:
        return False
    if lookahead_state.task is not None:
        return lookahead_state.scheduled_signature != main_state.signature
    artifact = lookahead_state.artifact
    return artifact is None or artifact.signature != main_state.signature


def should_commit(main_state: LookaheadMainState) -> bool:
    return main_state.ready and main_state.usage >= main_state.hard_limit
