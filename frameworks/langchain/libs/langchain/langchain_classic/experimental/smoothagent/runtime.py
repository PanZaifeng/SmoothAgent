"""Lookahead execution runtime.

This implements the runtime described by the paper's lookahead programming
model (``subsec:programming_model``):

- :meth:`LookaheadRuntime.on_segment_boundary` is invoked after every
  agent generate / tool observation append. It snapshots the current
  message list and starts an async transform task if the strategy's
  ``should_lookahead`` predicate is satisfied.
- :meth:`LookaheadRuntime.on_commit` is invoked just before the next LLM
  call. If the strategy says ``should_commit`` and the lookahead task has
  already finished, the runtime returns the transformed messages directly.
  Otherwise it tries a promotion (best-effort) and finally falls back to a
  synchronous transform.

All lookahead-stream LLM calls are expected to carry the
``LookaheadRequestMeta`` extra-body — see :mod:`request_meta`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Any

from langchain_classic.experimental.smoothagent.strategies import (
    LookaheadStrategy,
)

logger = logging.getLogger(__name__)


def _snapshot(messages: list[Any]) -> list[Any]:
    return list(messages)


def _message_payload(message: Any) -> Any:
    if isinstance(message, dict):
        return {str(k): _message_payload(v) for k, v in sorted(message.items())}
    if isinstance(message, (list, tuple)):
        return [_message_payload(item) for item in message]
    if isinstance(message, (str, int, float, bool)) or message is None:
        return message
    if hasattr(message, "model_dump"):
        try:
            return _message_payload(message.model_dump(mode="json"))
        except TypeError:
            return _message_payload(message.model_dump())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(message, "dict"):
        try:
            return _message_payload(message.dict())
        except Exception:  # noqa: BLE001
            pass
    return {
        "type": type(message).__name__,
        "role": getattr(message, "role", None),
        "content": getattr(message, "content", repr(message)),
    }


def message_signature(messages: list[Any]) -> str:
    """Return a stable signature for stale-artifact detection."""
    encoded = json.dumps(
        [_message_payload(message) for message in messages],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class LookaheadRuntime:
    """Drive a :class:`LookaheadStrategy` from an agent loop.

    The runtime is intentionally framework-neutral. Integrators call
    ``on_segment_boundary`` after every observation and ``on_commit``
    before every LLM generate.
    """

    def __init__(
        self,
        strategy: LookaheadStrategy,
        *,
        agent_id: str | None = None,
        promote_callback: "PromoteCallback | None" = None,
    ) -> None:
        self.strategy = strategy
        self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        self._task: asyncio.Task[list[Any]] | None = None
        self._task_id: str | None = None
        self._promote_callback = promote_callback
        self._task_started_at: float | None = None

    # ------------------------------------------------------------------
    # Segment boundary
    # ------------------------------------------------------------------

    async def on_segment_boundary(self, messages: list[Any]) -> None:
        """Refresh state and (optionally) launch a lookahead transform."""
        self.strategy.update_main(messages)
        self._log_event(
            "segment_boundary",
            messages=messages,
            should_lookahead=self.strategy.should_lookahead(messages),
        )
        if not self.strategy.should_lookahead(messages):
            return
        if self._task is not None and not self._task.done():
            return
        snapshot = _snapshot(messages)
        scheduled_signature = message_signature(snapshot)
        if (
            self.strategy.la.completed
            and self.strategy.la.completed_signature == scheduled_signature
        ):
            return
        task_id = uuid.uuid4().hex
        self._task_id = task_id
        self.strategy.la.pending_task_id = task_id
        self.strategy.la.scheduled_signature = scheduled_signature
        self.strategy.la.completed_signature = None
        self.strategy.la.completed = False
        self.strategy.la.error = None
        self._task_started_at = time.monotonic()
        self._task = asyncio.create_task(
            self._run_transform(snapshot, task_id, scheduled_signature)
        )

    async def _run_transform(
        self, messages: list[Any], task_id: str, scheduled_signature: str
    ) -> list[Any]:
        try:
            transformed = await self.strategy.transform(messages)
            segment_end = self.strategy.la.last_segment_end or len(messages)
            segment_end = min(segment_end, len(messages))
            self.strategy.la.transformed = list(transformed)
            self.strategy.la.last_segment_end = segment_end
            self.strategy.la.scheduled_signature = scheduled_signature
            self.strategy.la.completed_signature = message_signature(
                messages[:segment_end]
            )
            self.strategy.la.completed = True
            self._log_event(
                "lookahead_complete",
                messages=messages,
                task_id=task_id,
                transformed_len=len(transformed),
                duration_ms=self._task_duration_ms(),
            )
            return transformed
        except BaseException as exc:  # noqa: BLE001
            self.strategy.la.error = exc
            self.strategy.la.completed = False
            self._log_event(
                "lookahead_error",
                messages=messages,
                task_id=task_id,
                error=repr(exc),
                duration_ms=self._task_duration_ms(),
            )
            raise

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    async def on_commit(self, messages: list[Any]) -> list[Any]:
        """Return the transformed messages if commit should happen."""
        self.strategy.update_main(messages)
        if not self.strategy.should_commit(messages):
            return messages

        if self._task is not None and not self._task.done():
            if self.strategy.should_promote(messages):
                await self._promote_pending_lookahead()

        if self._task is not None:
            try:
                transformed = await self._task
            except BaseException:  # noqa: BLE001
                transformed = await self._sync_fallback(messages, fallback=True)
            else:
                committed = self._commit_completed_artifact(messages, transformed)
                if committed is None:
                    transformed = await self._sync_fallback(messages, fallback=True)
                    self._reset_after_commit()
                    return transformed
                self._log_event(
                    "commit",
                    messages=messages,
                    fallback_sync=False,
                    context_tokens_before=self.strategy.main.token_count,
                )
                self._reset_after_commit()
                return committed
            self._reset_after_commit()
            return transformed

        return await self._sync_fallback(messages, fallback=True)

    def _commit_completed_artifact(
        self, messages: list[Any], transformed: list[Any] | None = None
    ) -> list[Any] | None:
        segment_end = self.strategy.la.last_segment_end
        completed_signature = self.strategy.la.completed_signature
        if not self.strategy.la.completed or completed_signature is None:
            return None
        if segment_end > len(messages):
            self._log_event("stale_lookahead_artifact", messages=messages)
            return None
        current_signature = message_signature(messages[:segment_end])
        if current_signature != completed_signature:
            self._log_event(
                "stale_lookahead_artifact",
                messages=messages,
                reason="prefix_signature_mismatch",
            )
            return None
        artifact = list(transformed if transformed is not None else self.strategy.la.transformed)
        if segment_end == len(messages):
            return artifact
        if self.strategy.la.strategy_data.get("allow_suffix_commit") is True:
            return [*artifact, *messages[segment_end:]]
        self._log_event(
            "stale_lookahead_artifact",
            messages=messages,
            reason="suffix_not_supported",
        )
        return None

    async def _sync_fallback(
        self, messages: list[Any], *, fallback: bool
    ) -> list[Any]:
        before_tokens = self.strategy.main.token_count
        try:
            transformed = await self.strategy.transform(messages)
        except BaseException as exc:  # noqa: BLE001
            self._log_event(
                "fallback_error",
                messages=messages,
                error=repr(exc),
            )
            raise
        self._log_event(
            "commit",
            messages=messages,
            fallback_sync=fallback,
            context_tokens_before=before_tokens,
        )
        self._reset_after_commit()
        return transformed

    async def _promote_pending_lookahead(self) -> None:
        if self._promote_callback is None or self._task_id is None:
            return
        try:
            result = self._promote_callback(self._task_id)
            if asyncio.iscoroutine(result):
                await result
        except BaseException as exc:  # noqa: BLE001
            self._log_event(
                "promote_error",
                messages=self.strategy.main.messages,
                error=repr(exc),
            )

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _reset_after_commit(self) -> None:
        self._task = None
        self._task_id = None
        self._task_started_at = None
        self.strategy.la.reset()

    def _task_duration_ms(self) -> float | None:
        if self._task_started_at is None:
            return None
        return (time.monotonic() - self._task_started_at) * 1000.0

    def _log_event(
        self,
        event: str,
        *,
        messages: list[Any],
        **fields: Any,
    ) -> None:
        record: dict[str, Any] = {
            "event": event,
            "agent_id": self.agent_id,
            "strategy": type(self.strategy).__name__,
            "context_tokens": self.strategy.main.token_count,
            "turn": self.strategy.main.turn_count,
            "lookahead_task_id": self._task_id,
            "lookahead_completed": self.strategy.la.completed,
        }
        record.update(fields)
        record = {k: v for k, v in record.items() if not _is_messages(v)}
        try:
            logger.info(json.dumps(record, default=str))
        except (TypeError, ValueError):
            logger.info("%s", record)


def _is_messages(value: Any) -> bool:
    return isinstance(value, list) and value and not isinstance(value[0], (int, float, str, bool))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from collections.abc import Awaitable, Callable  # noqa: E402

PromoteCallback = Callable[[str], "Awaitable[Any] | Any"]
"""Callback invoked when a pending lookahead should have its priority bumped.

Receives the lookahead task id. Typically implemented as an HTTP / RPC call
to the SGLang ``/lookahead/control`` endpoint that promotes the matching
session task to the LC fast lane.
"""
