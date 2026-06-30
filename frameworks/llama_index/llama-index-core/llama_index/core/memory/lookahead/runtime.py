"""Runtime for scheduling and committing lookahead memory transforms."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

from llama_index.core.async_utils import asyncio_run
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory.lookahead.state import LookaheadArtifact
from llama_index.core.memory.lookahead.strategies import LookaheadStrategy

PromoteCallback = Callable[[str], Awaitable[Any] | Any]


def messages_signature(messages: list[ChatMessage]) -> str:
    """Return a stable signature for a message snapshot."""
    payload: list[Any] = []
    for message in messages:
        if hasattr(message, "model_dump"):
            payload.append(message.model_dump(mode="json"))
        else:
            payload.append(str(message))
    data = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class LookaheadRuntime:
    """Drive a ``LookaheadStrategy`` from memory segment and commit hooks."""

    def __init__(
        self,
        strategy: LookaheadStrategy,
        *,
        agent_id: Optional[str] = None,
        promote_callback: Optional[PromoteCallback] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self.strategy = strategy
        self.agent_id = agent_id or f"llamaindex_{uuid.uuid4().hex[:8]}"
        self._promote_callback = promote_callback
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="llamaindex-lookahead"
        )
        self._owns_executor = executor is None
        self._future: Optional[Future[LookaheadArtifact]] = None
        self._task_id: Optional[str] = None
        self._latest_signature: Optional[str] = None
        self._lock = threading.Lock()

    async def on_segment_boundary(self, messages: list[ChatMessage]) -> None:
        """Refresh state and schedule a background transform if needed."""
        snapshot = list(messages)
        self.strategy.update_main(snapshot)
        signature = messages_signature(snapshot)
        with self._lock:
            self._latest_signature = signature

        if not self.strategy.should_lookahead():
            return

        with self._lock:
            if self._future is not None and not self._future.done():
                return

            task_id = uuid.uuid4().hex
            self._task_id = task_id
            self.strategy.la.generation += 1
            generation = self.strategy.la.generation
            self.strategy.la.pending_task_id = task_id
            self.strategy.la.scheduled_signature = signature
            self.strategy.la.completed_signature = None
            self.strategy.la.completed = False
            self.strategy.la.artifact = None
            self.strategy.la.error = None
            self._future = self._executor.submit(
                self._run_transform_blocking,
                snapshot,
                signature,
                task_id,
                generation,
            )

    def on_segment_boundary_sync(self, messages: list[ChatMessage]) -> None:
        """Synchronous wrapper for ``on_segment_boundary``."""
        asyncio_run(self.on_segment_boundary(messages))

    async def on_commit(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Commit a matching artifact or run a synchronous fallback transform."""
        snapshot = list(messages)
        self.strategy.update_main(snapshot)
        signature = messages_signature(snapshot)
        with self._lock:
            self._latest_signature = signature
            future = self._future
            scheduled_signature = self.strategy.la.scheduled_signature

        if not self.strategy.should_commit():
            return messages

        if future is not None and scheduled_signature != signature:
            await self._cancel_stale_future(future)
            return await self._sync_fallback(snapshot)

        if future is not None:
            if not future.done() and self.strategy.should_promote():
                await self._promote_pending_lookahead()
            try:
                artifact = await asyncio.wrap_future(future)
            except BaseException:  # noqa: BLE001
                return await self._sync_fallback(snapshot)
            if self.can_commit_artifact(artifact, snapshot):
                return self.commit_artifact(artifact, snapshot)
            return await self._sync_fallback(snapshot)

        artifact = self.strategy.la.artifact
        if artifact is not None and self.can_commit_artifact(artifact, snapshot):
            return self.commit_artifact(artifact, snapshot)

        return await self._sync_fallback(snapshot)

    def on_commit_sync(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Synchronous wrapper for ``on_commit``."""
        return asyncio_run(self.on_commit(messages))

    def can_commit_artifact(
        self, artifact: LookaheadArtifact, messages: list[ChatMessage]
    ) -> bool:
        """Return whether ``artifact`` was built for ``messages``."""
        return artifact.source_signature == messages_signature(list(messages))

    def commit_artifact(
        self, artifact: LookaheadArtifact, messages: list[ChatMessage]
    ) -> list[ChatMessage]:
        """Commit a non-stale artifact, rejecting mismatched snapshots."""
        if not self.can_commit_artifact(artifact, messages):
            raise ValueError("Cannot commit stale lookahead artifact")
        transformed = list(artifact.transformed)
        self._reset_after_commit()
        return transformed

    def reset(self) -> None:
        """Cancel pending work and clear runtime state."""
        with self._lock:
            future = self._future
            self._future = None
            self._task_id = None
            self._latest_signature = None
        if future is not None and not future.done():
            future.cancel()
        self.strategy.la.reset()

    def shutdown(self) -> None:
        """Release the runtime executor when it is owned by this runtime."""
        self.reset()
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _run_transform_blocking(
        self,
        messages: list[ChatMessage],
        signature: str,
        task_id: str,
        generation: int,
    ) -> LookaheadArtifact:
        return asyncio.run(
            self._run_transform(messages, signature, task_id, generation)
        )

    async def _run_transform(
        self,
        messages: list[ChatMessage],
        signature: str,
        task_id: str,
        generation: int,
    ) -> LookaheadArtifact:
        try:
            transformed = await self.strategy.transform(messages)
        except BaseException as exc:  # noqa: BLE001
            with self._lock:
                if self.strategy.la.pending_task_id == task_id:
                    self.strategy.la.error = exc
                    self.strategy.la.completed = False
            raise

        artifact = LookaheadArtifact(
            transformed=list(transformed),
            source_signature=signature,
            task_id=task_id,
            generation=generation,
        )
        with self._lock:
            is_current_task = self.strategy.la.pending_task_id == task_id
            is_current_snapshot = self._latest_signature == signature
            if is_current_task and is_current_snapshot:
                self.strategy.la.transformed = list(transformed)
                self.strategy.la.completed = True
                self.strategy.la.completed_signature = signature
                self.strategy.la.artifact = artifact
                self.strategy.la.error = None
            elif is_current_task:
                self.strategy.la.completed = False
                self.strategy.la.completed_signature = None
                self.strategy.la.artifact = artifact
        return artifact

    async def _sync_fallback(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        try:
            transformed = await self.strategy.transform(messages)
        finally:
            self._reset_after_commit()
        return list(transformed)

    async def _cancel_stale_future(self, future: Future[LookaheadArtifact]) -> None:
        if not future.done():
            future.cancel()
        with self._lock:
            self._future = None
            self._task_id = None
            self.strategy.la.pending_task_id = None
            self.strategy.la.completed = False
            self.strategy.la.completed_signature = None

    async def _promote_pending_lookahead(self) -> None:
        if self._promote_callback is None or self._task_id is None:
            return
        result = self._promote_callback(self._task_id)
        if asyncio.iscoroutine(result):
            await result

    def _reset_after_commit(self) -> None:
        with self._lock:
            self._future = None
            self._task_id = None
        self.strategy.la.reset()
