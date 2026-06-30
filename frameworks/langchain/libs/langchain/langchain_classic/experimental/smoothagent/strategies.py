"""Lookahead context-engineering strategies.

This module provides the :class:`LookaheadStrategy` abstract base class from
``list:lookahead`` and three reference implementations: offloading,
sliding-window / keep-recent-K, and summarization.

The strategies operate on a generic ``messages`` sequence — anything that
provides ``role`` / ``content`` (langchain ``BaseMessage``, plain dicts, or
strings). They do not import from :mod:`langchain_classic.memory.lookahead`
or :mod:`langchain.agents.middleware.summarization`; both modules continue
to work unchanged. Use this module when you want the paper's four-method
contract (``should_lookahead`` / ``transform`` / ``should_commit`` /
``should_promote``).
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_classic.experimental.smoothagent.offload_store import (
    OffloadRecord,
    OffloadStore,
)
from langchain_classic.experimental.smoothagent.state import (
    LookaheadState,
    MainState,
)
from langchain_classic.experimental.smoothagent.token_utils import (
    Tokenizer,
    count_message_tokens,
    count_tokens,
)


def _role_of(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", message.get("type", ""))).lower()
    if hasattr(message, "type"):
        return str(getattr(message, "type")).lower()
    role = getattr(message, "role", None)
    return str(role).lower() if role else ""


def _is_observation(message: Any) -> bool:
    """Return ``True`` if ``message`` looks like a tool / observation block."""
    role = _role_of(message)
    if role in {"tool", "function", "observation", "tool_message"}:
        return True
    name = getattr(message, "__class__", type(message)).__name__.lower()
    return "tool" in name or "observation" in name


def _is_system(message: Any) -> bool:
    return _role_of(message) in {"system", "system_message"}


class LookaheadStrategy(ABC):
    """Abstract contract for a lookahead context-engineering strategy.

    All four predicates are intentionally synchronous — they observe state
    that is already known to the runtime. Only :meth:`transform` is allowed
    to be asynchronous because it may issue LLM calls.
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.main = MainState()
        self.la = LookaheadState()
        self._tokenizer = tokenizer

    def update_main(self, messages: list[Any]) -> None:
        """Refresh :attr:`main` from the live message list."""
        self.main.messages = list(messages)
        self.main.token_count = count_tokens(messages, self._tokenizer)
        self.main.turn_count = sum(
            1 for m in messages if _role_of(m) in {"human", "user", "ai", "assistant"}
        )

    @abstractmethod
    def should_lookahead(self, messages: list[Any]) -> bool:
        """Return ``True`` if a new lookahead transform should be queued."""

    @abstractmethod
    async def transform(self, messages: list[Any]) -> list[Any]:
        """Compute the transformed messages from ``messages``.

        Implementations MUST depend only on ``messages`` and on
        :attr:`la.strategy_data`. They MUST NOT depend on any token
        produced after ``messages``.
        """

    @abstractmethod
    def should_commit(self, messages: list[Any]) -> bool:
        """Return ``True`` if the transformed prefix should be committed."""

    def should_promote(self, messages: list[Any]) -> bool:
        """Return ``True`` if the pending lookahead should be promoted."""
        return self.should_commit(messages)


# ---------------------------------------------------------------------------
# Offloading
# ---------------------------------------------------------------------------


@dataclass
class OffloadConfig:
    hard_token_limit: int = 15000
    min_observation_tokens: int = 1024
    offload_dir: str = "/tmp/smoothagent_offload"
    preserve_recent_observations: int = 1


class OffloadLookaheadStrategy(LookaheadStrategy):
    """Offload long observations to a filesystem store.

    The transform writes any observation longer than
    ``config.min_observation_tokens`` into ``config.offload_dir`` and
    replaces it in the message list with a short reference placeholder.
    Observations that have already been offloaded are skipped on
    subsequent calls — :attr:`la.strategy_data["offloaded"]` records
    the message indices that have been processed.
    """

    def __init__(
        self,
        config: OffloadConfig | None = None,
        *,
        store: OffloadStore | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(tokenizer=tokenizer)
        self.config = config or OffloadConfig()
        self.store = store or OffloadStore(self.config.offload_dir)
        self.la.strategy_data.setdefault("offloaded", {})

    def _offload_targets(self, messages: list[Any]) -> list[int]:
        already: dict[int, OffloadRecord] = self.la.strategy_data.get("offloaded", {})
        observation_indices = [
            i for i, m in enumerate(messages) if _is_observation(m)
        ]
        keep_recent = max(0, self.config.preserve_recent_observations)
        if keep_recent > 0:
            observation_indices = observation_indices[:-keep_recent] if len(
                observation_indices
            ) > keep_recent else []
        return [
            i
            for i in observation_indices
            if i not in already
            and count_message_tokens(messages[i], self._tokenizer)
            >= self.config.min_observation_tokens
        ]

    def should_lookahead(self, messages: list[Any]) -> bool:
        return bool(self._offload_targets(messages))

    async def transform(self, messages: list[Any]) -> list[Any]:
        await asyncio.sleep(0)  # yield control, transform is fast
        already: dict[int, OffloadRecord] = self.la.strategy_data.setdefault(
            "offloaded", {}
        )
        transformed = list(messages)
        for index in self._offload_targets(messages):
            text = self._observation_text(messages[index])
            original = count_message_tokens(messages[index], self._tokenizer)
            placeholder_record = self.store.store(
                text,
                observation_index=index,
                original_token_count=original,
                reference_token_count=0,
            )
            reference = self.store.render_reference(placeholder_record)
            ref_tokens = count_message_tokens(reference, self._tokenizer)
            record = OffloadRecord(
                observation_id=placeholder_record.observation_id,
                file_path=placeholder_record.file_path,
                original_token_count=original,
                reference_token_count=ref_tokens,
            )
            already[index] = record
            transformed[index] = self._make_reference_message(
                messages[index], reference
            )
        self.la.last_segment_end = len(messages)
        return transformed

    def should_commit(self, messages: list[Any]) -> bool:
        return count_tokens(messages, self._tokenizer) >= self.config.hard_token_limit

    @staticmethod
    def _observation_text(message: Any) -> str:
        if isinstance(message, str):
            return message
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(getattr(message, "content", message))

    @staticmethod
    def _make_reference_message(original: Any, reference: str) -> Any:
        if isinstance(original, str):
            return reference
        if isinstance(original, dict):
            new = dict(original)
            new["content"] = reference
            return new
        if hasattr(original, "model_copy"):
            try:
                return original.model_copy(update={"content": reference})
            except Exception:  # noqa: BLE001
                pass
        if hasattr(original, "copy"):
            try:
                return original.copy(update={"content": reference})
            except Exception:  # noqa: BLE001
                pass
        try:
            original.content = reference
        except Exception:  # noqa: BLE001
            return reference
        return original


# ---------------------------------------------------------------------------
# Sliding window / keep-recent-K
# ---------------------------------------------------------------------------


@dataclass
class SlidingWindowConfig:
    """Configuration for turn- and token-based sliding-window projection.

    ``max_turns`` remains the retained turn-message count for backward
    compatibility. New soft/hard limits control when lookahead is queued
    and when the transformed context must be committed.
    """

    max_turns: int = 10
    preserve_system_prompt: bool = True
    soft_turn_limit: int | None = None
    hard_turn_limit: int | None = None
    soft_token_limit: int | None = None
    hard_token_limit: int | None = None
    keep_recent_tokens: int | None = None


class SlidingWindowLookaheadStrategy(LookaheadStrategy):
    """Keep only the most recent ``max_turns`` turns plus the system prompt."""

    def __init__(
        self,
        config: SlidingWindowConfig | None = None,
        *,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(tokenizer=tokenizer)
        self.config = config or SlidingWindowConfig()

    def _turns(self, messages: list[Any]) -> list[int]:
        return [
            i
            for i, m in enumerate(messages)
            if _role_of(m) in {"human", "user", "ai", "assistant"}
        ]

    @property
    def _hard_turn_limit(self) -> int:
        return self.config.hard_turn_limit or self.config.max_turns

    @property
    def _soft_turn_limit(self) -> int:
        return self.config.soft_turn_limit or self._hard_turn_limit

    def should_lookahead(self, messages: list[Any]) -> bool:
        if len(self._turns(messages)) > self._soft_turn_limit:
            return True
        token_limit = self.config.soft_token_limit or self.config.hard_token_limit
        if token_limit is not None:
            return count_tokens(messages, self._tokenizer) > token_limit
        return False

    async def transform(self, messages: list[Any]) -> list[Any]:
        await asyncio.sleep(0)
        transformed = self._apply_turn_window(messages)
        transformed = self._apply_token_window(transformed)
        self.la.last_segment_end = len(messages)
        self.la.strategy_data["dropped_count"] = max(
            0, len(messages) - len(transformed)
        )
        return transformed

    def _apply_turn_window(self, messages: list[Any]) -> list[Any]:
        turns = self._turns(messages)
        if len(turns) <= self.config.max_turns:
            return list(messages)

        keep_from = turns[-self.config.max_turns]
        kept: list[Any] = []
        if self.config.preserve_system_prompt:
            kept.extend(m for m in messages[:keep_from] if _is_system(m))
        kept.extend(messages[keep_from:])
        return kept

    def _apply_token_window(self, messages: list[Any]) -> list[Any]:
        token_budget = self.config.keep_recent_tokens or self.config.hard_token_limit
        if token_budget is None:
            return list(messages)
        if count_tokens(messages, self._tokenizer) <= token_budget:
            return list(messages)

        system: list[Any] = []
        body = list(messages)
        if self.config.preserve_system_prompt and body and _is_system(body[0]):
            system = [body[0]]
            body = body[1:]

        kept: list[Any] = []
        kept_tokens = count_tokens(system, self._tokenizer)
        for message in reversed(body):
            message_tokens = count_message_tokens(message, self._tokenizer)
            if kept and kept_tokens + message_tokens > token_budget:
                break
            if not kept and kept_tokens + message_tokens > token_budget:
                break
            kept.insert(0, message)
            kept_tokens += message_tokens
        return [*system, *kept]

    def should_commit(self, messages: list[Any]) -> bool:
        if len(self._turns(messages)) > self._hard_turn_limit:
            return True
        if self.config.hard_token_limit is not None:
            return count_tokens(messages, self._tokenizer) > self.config.hard_token_limit
        return False


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


SummaryCallable = Callable[[list[Any]], "Awaitable[str] | str"]
"""Either ``async def(messages) -> str`` or ``def(messages) -> str``."""


SUMMARIZATION_PROMPT = (
    "You are compressing the previous context of a long-running LLM agent.\n"
    "Summarize only the information that may be needed for future reasoning.\n"
    "Preserve:\n"
    "- user goals and constraints;\n"
    "- files, tools, commands, and observations already inspected;\n"
    "- intermediate conclusions;\n"
    "- unresolved questions;\n"
    "- decisions already made.\n"
    "Do not include irrelevant wording.\n"
    "Keep the summary concise and factual."
)


@dataclass
class SummarizationConfig:
    soft_token_limit: int = 11000
    hard_token_limit: int = 15000
    recent_token_budget: int = 4000
    summary_max_tokens: int = 128
    summary_model: str | None = None
    summary_role: str = "system"
    prompt: str = SUMMARIZATION_PROMPT
    summary_strategy_data_key: str = "summary"


class SummarizationLookaheadStrategy(LookaheadStrategy):
    """Replace older turns with an LLM-generated summary at the soft limit.

    The strategy delegates the actual LLM call to ``summary_callable``,
    which keeps this module free of hard dependencies on a specific model
    client. Any LLM call inside ``summary_callable`` should be marked as
    a best-effort lookahead request — see
    :func:`langchain_classic.experimental.smoothagent.request_meta.build_extra_body`.
    """

    def __init__(
        self,
        summary_callable: SummaryCallable,
        config: SummarizationConfig | None = None,
        *,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(tokenizer=tokenizer)
        self.config = config or SummarizationConfig()
        self._summary_callable = summary_callable

    def should_lookahead(self, messages: list[Any]) -> bool:
        return (
            count_tokens(messages, self._tokenizer) >= self.config.soft_token_limit
            and self.la.pending_task_id is None
            and not self.la.completed
        )

    def should_commit(self, messages: list[Any]) -> bool:
        return count_tokens(messages, self._tokenizer) >= self.config.hard_token_limit

    def _split(self, messages: list[Any]) -> tuple[list[Any], list[Any], list[Any]]:
        system: list[Any] = []
        body: list[Any] = []
        for m in messages:
            if _is_system(m) and not body:
                system.append(m)
            else:
                body.append(m)

        recent: list[Any] = []
        recent_tokens = 0
        for m in reversed(body):
            tokens = count_message_tokens(m, self._tokenizer)
            if recent and recent_tokens + tokens > self.config.recent_token_budget:
                break
            recent.insert(0, m)
            recent_tokens += tokens
        old = body[: len(body) - len(recent)]
        return system, old, recent

    async def transform(self, messages: list[Any]) -> list[Any]:
        system, old, recent = self._split(messages)
        if not old:
            self.la.last_segment_end = len(messages)
            return list(messages)

        summary_prompt_messages: list[Any] = [
            {"role": "system", "content": self.config.prompt},
            *old,
        ]
        result = self._summary_callable(summary_prompt_messages)
        if inspect.isawaitable(result):
            summary_text = await result
        else:
            summary_text = result  # type: ignore[assignment]
        summary_text = str(summary_text).strip() or "(no summary produced)"
        summary_message = {
            "role": self.config.summary_role,
            "content": f"Summary of earlier turns:\n{summary_text}",
        }
        self.la.strategy_data[self.config.summary_strategy_data_key] = summary_text
        self.la.last_segment_end = len(messages)
        return [*system, summary_message, *recent]
