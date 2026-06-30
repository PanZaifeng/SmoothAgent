"""Lookahead context-engineering strategies for ``ChatMessage`` memory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.lookahead.request_meta import (
    LookaheadRequestMeta,
    build_extra_body,
)
from llama_index.core.memory.lookahead.state import LookaheadState, MainState
from llama_index.core.utils import get_tokenizer

TokenizerFn = Callable[[str], list[Any]]

DEFAULT_SUMMARIZE_PROMPT = (
    "The following is a conversation between the user and assistant. "
    "Write a concise summary about the contents of this conversation."
)


def token_count_for_messages(
    messages: list[ChatMessage], tokenizer_fn: TokenizerFn
) -> int:
    """Count message content tokens the same way as ``ChatMemoryBuffer``."""
    if not messages:
        return 0
    msg_str = " ".join(str(message.content) for message in messages)
    return len(tokenizer_fn(msg_str))


def trim_chat_messages_by_token_limit(
    chat_history: list[ChatMessage],
    *,
    token_limit: int,
    tokenizer_fn: TokenizerFn,
    initial_token_count: int = 0,
) -> list[ChatMessage]:
    """Return the latest messages that fit ``token_limit``.

    This intentionally mirrors ``ChatMemoryBuffer.get`` so the explicit
    lookahead buffer can preserve synchronous memory behavior.
    """
    if initial_token_count > token_limit:
        raise ValueError("Initial token count exceeds token limit")
    if not chat_history:
        return []

    message_count = len(chat_history)
    cur_messages = chat_history[-message_count:]
    token_count = (
        token_count_for_messages(cur_messages, tokenizer_fn) + initial_token_count
    )

    while token_count > token_limit and message_count > 1:
        message_count -= 1
        while message_count > 1 and chat_history[-message_count].role in (
            MessageRole.TOOL,
            MessageRole.ASSISTANT,
        ):
            message_count -= 1

        cur_messages = chat_history[-message_count:]
        token_count = (
            token_count_for_messages(cur_messages, tokenizer_fn) + initial_token_count
        )

    if token_count > token_limit or message_count <= 0:
        return chat_history[-1:]

    return chat_history[-message_count:]


def _role_value(role: Any) -> str:
    return str(role.value) if hasattr(role, "value") else str(role)


def _turn_count(messages: list[ChatMessage]) -> int:
    return sum(
        1
        for message in messages
        if message.role in (MessageRole.USER, MessageRole.ASSISTANT)
    )


class LookaheadStrategy(ABC):
    """Strategy contract for lookahead memory transforms."""

    def __init__(self, *, tokenizer_fn: Optional[TokenizerFn] = None) -> None:
        self.main = MainState()
        self.la = LookaheadState()
        self.tokenizer_fn = tokenizer_fn or get_tokenizer()

    def update_main(self, messages: list[ChatMessage]) -> None:
        """Refresh foreground state from the live message list."""
        self.main.messages = list(messages)
        self.main.token_count = token_count_for_messages(
            self.main.messages, self.tokenizer_fn
        )
        self.main.turn_count = _turn_count(self.main.messages)

    @abstractmethod
    def should_lookahead(self) -> bool:
        """Return whether a background transform should be scheduled."""

    @abstractmethod
    async def transform(
        self, messages: Optional[list[ChatMessage]] = None
    ) -> list[ChatMessage]:
        """Transform ``messages`` or the current main-state snapshot."""

    @abstractmethod
    def should_commit(self) -> bool:
        """Return whether a completed transform should be committed."""

    def should_promote(self) -> bool:
        """Return whether pending lookahead work should be promoted."""
        return self.should_commit()


@dataclass
class SlidingWindowLookaheadConfig:
    """Configuration for sliding-window lookahead memory."""

    token_limit: int = 3000
    soft_token_limit: Optional[int] = None
    hard_token_limit: Optional[int] = None
    initial_token_count: int = 0


class SlidingWindowLookaheadStrategy(LookaheadStrategy):
    """Precompute the same token window returned by ``ChatMemoryBuffer``."""

    def __init__(
        self,
        config: Optional[SlidingWindowLookaheadConfig] = None,
        *,
        tokenizer_fn: Optional[TokenizerFn] = None,
    ) -> None:
        super().__init__(tokenizer_fn=tokenizer_fn)
        self.config = config or SlidingWindowLookaheadConfig()

    @property
    def initial_token_count(self) -> int:
        return self.config.initial_token_count

    @initial_token_count.setter
    def initial_token_count(self, value: int) -> None:
        self.config.initial_token_count = value

    @property
    def soft_token_limit(self) -> int:
        return self.config.soft_token_limit or self.config.token_limit

    @property
    def hard_token_limit(self) -> int:
        return self.config.hard_token_limit or self.config.token_limit

    def should_lookahead(self) -> bool:
        return (
            self.main.token_count + self.initial_token_count >= self.soft_token_limit
            and self.la.pending_task_id is None
            and not self.la.completed
        )

    async def transform(
        self, messages: Optional[list[ChatMessage]] = None
    ) -> list[ChatMessage]:
        snapshot = list(messages if messages is not None else self.main.messages)
        transformed = trim_chat_messages_by_token_limit(
            snapshot,
            token_limit=self.config.token_limit,
            tokenizer_fn=self.tokenizer_fn,
            initial_token_count=self.initial_token_count,
        )
        self.la.last_segment_end = len(snapshot)
        return transformed

    def should_commit(self) -> bool:
        return self.main.token_count + self.initial_token_count >= self.hard_token_limit


@dataclass
class SummarizationLookaheadConfig:
    """Configuration for summarization lookahead memory."""

    token_limit: int = 2000
    soft_token_limit: Optional[int] = None
    hard_token_limit: Optional[int] = None
    recent_token_budget: Optional[int] = None
    initial_token_count: int = 0
    count_initial_tokens: bool = False
    summarize_prompt: str = DEFAULT_SUMMARIZE_PROMPT
    summary_extra_body: Optional[dict[str, Any]] = None
    summary_llm_kwargs: Optional[dict[str, Any]] = None
    mark_sglang_lookahead: bool = True


class SummarizationLookaheadStrategy(LookaheadStrategy):
    """Summarize older chat messages while preserving a recent tail."""

    def __init__(
        self,
        llm: Optional[LLM],
        config: Optional[SummarizationLookaheadConfig] = None,
        *,
        tokenizer_fn: Optional[TokenizerFn] = None,
    ) -> None:
        super().__init__(tokenizer_fn=tokenizer_fn)
        self.llm = llm
        self.config = config or SummarizationLookaheadConfig()
        if (
            self.config.summary_extra_body is None
            and self.config.mark_sglang_lookahead
        ):
            self.config.summary_extra_body = build_extra_body(
                LookaheadRequestMeta(
                    is_lookahead=True, request_class="bg", priority_class="be"
                )
            )

    @property
    def initial_token_count(self) -> int:
        if not self.config.count_initial_tokens:
            return 0
        return self.config.initial_token_count

    @initial_token_count.setter
    def initial_token_count(self, value: int) -> None:
        self.config.initial_token_count = value

    @property
    def soft_token_limit(self) -> int:
        return self.config.soft_token_limit or self.hard_token_limit

    @property
    def hard_token_limit(self) -> int:
        return self.config.hard_token_limit or self.config.token_limit

    @property
    def recent_token_budget(self) -> int:
        return self.config.recent_token_budget or self.config.token_limit

    def should_lookahead(self) -> bool:
        return (
            self.main.token_count + self.initial_token_count >= self.soft_token_limit
            and self.la.pending_task_id is None
            and not self.la.completed
        )

    def should_commit(self) -> bool:
        if self.initial_token_count > self.recent_token_budget:
            return True
        return self.main.token_count + self.initial_token_count >= self.hard_token_limit

    async def transform(
        self, messages: Optional[list[ChatMessage]] = None
    ) -> list[ChatMessage]:
        snapshot = list(messages if messages is not None else self.main.messages)
        if self.initial_token_count > self.recent_token_budget:
            raise ValueError("Initial token count exceeds token limit")
        if not snapshot:
            return []

        full_text, to_summarize = self._split_messages_summary_or_full_text(snapshot)
        if self.llm is None or not to_summarize:
            updated_history = full_text
        else:
            updated_history = [
                await self._summarize_oldest_chat_history(to_summarize),
                *full_text,
            ]

        self.la.last_segment_end = len(snapshot)
        return updated_history

    def _split_messages_summary_or_full_text(
        self, chat_history: list[ChatMessage]
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        chat_history_full_text: list[ChatMessage] = []
        remaining = list(chat_history)

        while (
            remaining
            and self.initial_token_count
            + token_count_for_messages(chat_history_full_text, self.tokenizer_fn)
            + token_count_for_messages([remaining[-1]], self.tokenizer_fn)
            <= self.recent_token_budget
        ):
            chat_history_full_text.insert(0, remaining.pop())

        chat_history_to_be_summarized = remaining.copy()
        self._handle_assistant_and_tool_messages(
            chat_history_full_text, chat_history_to_be_summarized
        )
        return chat_history_full_text, chat_history_to_be_summarized

    async def _summarize_oldest_chat_history(
        self, chat_history_to_be_summarized: list[ChatMessage]
    ) -> ChatMessage:
        assert self.llm is not None

        if (
            len(chat_history_to_be_summarized) == 1
            and chat_history_to_be_summarized[0].role == MessageRole.SYSTEM
        ):
            return chat_history_to_be_summarized[0]

        summarize_prompt = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=self.config.summarize_prompt,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=self._get_prompt_to_summarize(chat_history_to_be_summarized),
            ),
        ]
        response = await self.llm.achat(summarize_prompt, **self._summary_call_kwargs())
        summary_text = self._summary_text_from_response(response)
        self.la.strategy_data["summary"] = summary_text
        return ChatMessage(role=MessageRole.SYSTEM, content=summary_text)

    def _summary_call_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.config.summary_llm_kwargs or {})
        extra_body = dict(self.config.summary_extra_body or {})
        existing_extra_body = kwargs.get("extra_body")
        if existing_extra_body is not None and not isinstance(existing_extra_body, dict):
            raise ValueError("summary_llm_kwargs['extra_body'] must be a dict")
        if extra_body or existing_extra_body:
            kwargs["extra_body"] = {**extra_body, **(existing_extra_body or {})}
        return kwargs

    @staticmethod
    def _summary_text_from_response(response: ChatResponse) -> str:
        return str(response.message.content or "")

    @staticmethod
    def _get_prompt_to_summarize(
        chat_history_to_be_summarized: list[ChatMessage],
    ) -> str:
        prompt = '"Transcript so far: '
        for message in chat_history_to_be_summarized:
            if not isinstance(message.content, str):
                continue

            prompt += _role_value(message.role) + ": "
            if message.content:
                prompt += message.content + "\n\n"
            else:
                prompt += (
                    "\n".join(
                        [
                            f"Calling a function: {call!s}"
                            for call in message.additional_kwargs.get("tool_calls", [])
                        ]
                    )
                    + "\n\n"
                )
        prompt += '"\n\n'
        return prompt

    @staticmethod
    def _handle_assistant_and_tool_messages(
        chat_history_full_text: list[ChatMessage],
        chat_history_to_be_summarized: list[ChatMessage],
    ) -> None:
        while chat_history_full_text and chat_history_full_text[0].role in (
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
        ):
            chat_history_to_be_summarized.append(chat_history_full_text.pop(0))
