"""Explicit lookahead memory classes for LlamaIndex chat memory."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.chat_memory_buffer import (
    DEFAULT_TOKEN_LIMIT as DEFAULT_CHAT_TOKEN_LIMIT,
)
from llama_index.core.memory.chat_memory_buffer import (
    DEFAULT_TOKEN_LIMIT_RATIO as DEFAULT_CHAT_TOKEN_LIMIT_RATIO,
)
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.memory.chat_summary_memory_buffer import (
    DEFAULT_TOKEN_LIMIT as DEFAULT_SUMMARY_TOKEN_LIMIT,
)
from llama_index.core.memory.chat_summary_memory_buffer import (
    DEFAULT_TOKEN_LIMIT_RATIO as DEFAULT_SUMMARY_TOKEN_LIMIT_RATIO,
)
from llama_index.core.memory.chat_summary_memory_buffer import (
    SUMMARIZE_PROMPT,
    ChatSummaryMemoryBuffer,
)
from llama_index.core.memory.lookahead.runtime import LookaheadRuntime
from llama_index.core.memory.lookahead.strategies import (
    SlidingWindowLookaheadConfig,
    SlidingWindowLookaheadStrategy,
    SummarizationLookaheadConfig,
    SummarizationLookaheadStrategy,
)
from llama_index.core.memory.types import DEFAULT_CHAT_STORE_KEY
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.core.utils import get_tokenizer


class LookaheadChatMemoryBuffer(ChatMemoryBuffer):
    """Opt-in ``ChatMemoryBuffer`` variant with lookahead sliding-window commit."""

    soft_token_limit: Optional[int] = Field(default=None)
    hard_token_limit: Optional[int] = Field(default=None)
    lookahead_agent_id: Optional[str] = Field(default=None)

    _lookahead_strategy: SlidingWindowLookaheadStrategy = PrivateAttr()
    _lookahead_runtime: LookaheadRuntime = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "LookaheadChatMemoryBuffer"

    def model_post_init(self, __context: Any) -> None:
        self._configure_lookahead()

    @classmethod
    def from_defaults(
        cls,
        chat_history: Optional[list[ChatMessage]] = None,
        llm: Optional[LLM] = None,
        chat_store: Optional[BaseChatStore] = None,
        chat_store_key: str = DEFAULT_CHAT_STORE_KEY,
        token_limit: Optional[int] = None,
        tokenizer_fn: Optional[Callable[[str], list[Any]]] = None,
        soft_token_limit: Optional[int] = None,
        hard_token_limit: Optional[int] = None,
        lookahead_agent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> "LookaheadChatMemoryBuffer":
        """Create a lookahead chat memory buffer from defaults."""
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")

        if llm is not None:
            context_window = llm.metadata.context_window
            token_limit = token_limit or int(
                context_window * DEFAULT_CHAT_TOKEN_LIMIT_RATIO
            )
        elif token_limit is None:
            token_limit = DEFAULT_CHAT_TOKEN_LIMIT

        if chat_history is not None:
            chat_store = chat_store or SimpleChatStore()
            chat_store.set_messages(chat_store_key, chat_history)

        return cls(
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn or get_tokenizer(),
            chat_store=chat_store or SimpleChatStore(),
            chat_store_key=chat_store_key,
            soft_token_limit=soft_token_limit,
            hard_token_limit=hard_token_limit,
            lookahead_agent_id=lookahead_agent_id,
        )

    def get(
        self, input: Optional[str] = None, initial_token_count: int = 0, **kwargs: Any
    ) -> list[ChatMessage]:
        """Get chat history, committing a ready lookahead artifact at hard trigger."""
        self._lookahead_strategy.initial_token_count = initial_token_count
        history = self.get_all()
        transformed = self._lookahead_runtime.on_commit_sync(history)
        if transformed is history:
            return ChatMemoryBuffer.get(
                self,
                input=input,
                initial_token_count=initial_token_count,
                **kwargs,
            )
        return transformed

    async def aget(
        self, input: Optional[str] = None, initial_token_count: int = 0, **kwargs: Any
    ) -> list[ChatMessage]:
        """Get chat history asynchronously."""
        self._lookahead_strategy.initial_token_count = initial_token_count
        history = await self.aget_all()
        transformed = await self._lookahead_runtime.on_commit(history)
        if transformed is history:
            return await asyncio.to_thread(
                ChatMemoryBuffer.get,
                self,
                input=input,
                initial_token_count=initial_token_count,
                **kwargs,
            )
        return transformed

    def put(self, message: ChatMessage) -> None:
        """Put chat history and schedule lookahead if the soft trigger is met."""
        ChatMemoryBuffer.put(self, message)
        self._lookahead_runtime.on_segment_boundary_sync(self.get_all())

    async def aput(self, message: ChatMessage) -> None:
        """Put chat history asynchronously."""
        await ChatMemoryBuffer.aput(self, message)
        await self._lookahead_runtime.on_segment_boundary(await self.aget_all())

    def put_messages(self, messages: list[ChatMessage]) -> None:
        """Put multiple messages and schedule lookahead once."""
        for message in messages:
            ChatMemoryBuffer.put(self, message)
        self._lookahead_runtime.on_segment_boundary_sync(self.get_all())

    async def aput_messages(self, messages: list[ChatMessage]) -> None:
        """Put multiple messages asynchronously."""
        for message in messages:
            await ChatMemoryBuffer.aput(self, message)
        await self._lookahead_runtime.on_segment_boundary(await self.aget_all())

    def set(self, messages: list[ChatMessage]) -> None:
        """Set chat history."""
        ChatMemoryBuffer.set(self, messages)
        self._lookahead_runtime.reset()

    async def aset(self, messages: list[ChatMessage]) -> None:
        """Set chat history asynchronously."""
        await ChatMemoryBuffer.aset(self, messages)
        self._lookahead_runtime.reset()

    def reset(self) -> None:
        """Reset chat history."""
        ChatMemoryBuffer.reset(self)
        self._lookahead_runtime.reset()

    async def areset(self) -> None:
        """Reset chat history asynchronously."""
        await ChatMemoryBuffer.areset(self)
        self._lookahead_runtime.reset()

    def _configure_lookahead(self) -> None:
        config = SlidingWindowLookaheadConfig(
            token_limit=self.token_limit,
            soft_token_limit=self.soft_token_limit,
            hard_token_limit=self.hard_token_limit,
        )
        self._lookahead_strategy = SlidingWindowLookaheadStrategy(
            config=config, tokenizer_fn=self.tokenizer_fn
        )
        self._lookahead_runtime = LookaheadRuntime(
            self._lookahead_strategy, agent_id=self.lookahead_agent_id
        )


class LookaheadChatSummaryMemoryBuffer(ChatSummaryMemoryBuffer):
    """Opt-in ``ChatSummaryMemoryBuffer`` variant with lookahead summarization."""

    soft_token_limit: Optional[int] = Field(default=None)
    hard_token_limit: Optional[int] = Field(default=None)
    recent_token_budget: Optional[int] = Field(default=None)
    summary_extra_body: Optional[dict[str, Any]] = Field(default=None)
    summary_llm_kwargs: dict[str, Any] = Field(default_factory=dict)
    mark_sglang_lookahead: bool = Field(default=True)
    lookahead_agent_id: Optional[str] = Field(default=None)

    _lookahead_strategy: SummarizationLookaheadStrategy = PrivateAttr()
    _lookahead_runtime: LookaheadRuntime = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "LookaheadChatSummaryMemoryBuffer"

    def model_post_init(self, __context: Any) -> None:
        self._configure_lookahead()

    @classmethod
    def from_defaults(
        cls,
        chat_history: Optional[list[ChatMessage]] = None,
        llm: Optional[LLM] = None,
        chat_store: Optional[BaseChatStore] = None,
        chat_store_key: str = DEFAULT_CHAT_STORE_KEY,
        token_limit: Optional[int] = None,
        tokenizer_fn: Optional[Callable[[str], list[Any]]] = None,
        summarize_prompt: Optional[str] = None,
        count_initial_tokens: bool = False,
        soft_token_limit: Optional[int] = None,
        hard_token_limit: Optional[int] = None,
        recent_token_budget: Optional[int] = None,
        summary_extra_body: Optional[dict[str, Any]] = None,
        summary_llm_kwargs: Optional[dict[str, Any]] = None,
        mark_sglang_lookahead: bool = True,
        lookahead_agent_id: Optional[str] = None,
        **kwargs: Any,
    ) -> "LookaheadChatSummaryMemoryBuffer":
        """Create a lookahead chat summary memory buffer from defaults."""
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}")

        if llm is not None:
            context_window = llm.metadata.context_window
            token_limit = token_limit or int(
                context_window * DEFAULT_SUMMARY_TOKEN_LIMIT_RATIO
            )
        elif token_limit is None:
            token_limit = DEFAULT_SUMMARY_TOKEN_LIMIT

        chat_store = chat_store or SimpleChatStore()
        if chat_history is not None:
            chat_store.set_messages(chat_store_key, chat_history)

        return cls(
            llm=llm,
            token_limit=token_limit,
            tokenizer_fn=tokenizer_fn or get_tokenizer(),
            summarize_prompt=summarize_prompt or SUMMARIZE_PROMPT,
            chat_store=chat_store,
            chat_store_key=chat_store_key,
            count_initial_tokens=count_initial_tokens,
            soft_token_limit=soft_token_limit,
            hard_token_limit=hard_token_limit,
            recent_token_budget=recent_token_budget,
            summary_extra_body=summary_extra_body,
            summary_llm_kwargs=summary_llm_kwargs or {},
            mark_sglang_lookahead=mark_sglang_lookahead,
            lookahead_agent_id=lookahead_agent_id,
        )

    def get(
        self, input: Optional[str] = None, initial_token_count: int = 0, **kwargs: Any
    ) -> list[ChatMessage]:
        """Get chat history, committing summary lookahead at hard trigger."""
        self._lookahead_strategy.initial_token_count = initial_token_count
        history = self.get_all()
        transformed = self._lookahead_runtime.on_commit_sync(history)
        if transformed is history:
            return ChatSummaryMemoryBuffer.get(
                self,
                input=input,
                initial_token_count=initial_token_count,
                **kwargs,
            )
        self._replace_history(transformed)
        return transformed

    async def aget(
        self, input: Optional[str] = None, initial_token_count: int = 0, **kwargs: Any
    ) -> list[ChatMessage]:
        """Get chat history asynchronously."""
        self._lookahead_strategy.initial_token_count = initial_token_count
        history = self.get_all()
        transformed = await self._lookahead_runtime.on_commit(history)
        if transformed is history:
            return await asyncio.to_thread(
                ChatSummaryMemoryBuffer.get,
                self,
                input=input,
                initial_token_count=initial_token_count,
                **kwargs,
            )
        await asyncio.to_thread(self._replace_history, transformed)
        return transformed

    def put(self, message: ChatMessage) -> None:
        """Put chat history and schedule lookahead if the soft trigger is met."""
        ChatSummaryMemoryBuffer.put(self, message)
        self._lookahead_runtime.on_segment_boundary_sync(self.get_all())

    async def aput(self, message: ChatMessage) -> None:
        """Put chat history asynchronously."""
        await ChatSummaryMemoryBuffer.aput(self, message)
        await self._lookahead_runtime.on_segment_boundary(self.get_all())

    def put_messages(self, messages: list[ChatMessage]) -> None:
        """Put multiple messages and schedule lookahead once."""
        for message in messages:
            ChatSummaryMemoryBuffer.put(self, message)
        self._lookahead_runtime.on_segment_boundary_sync(self.get_all())

    async def aput_messages(self, messages: list[ChatMessage]) -> None:
        """Put multiple messages asynchronously."""
        for message in messages:
            await ChatSummaryMemoryBuffer.aput(self, message)
        await self._lookahead_runtime.on_segment_boundary(self.get_all())

    def set(self, messages: list[ChatMessage]) -> None:
        """Set chat history."""
        ChatSummaryMemoryBuffer.set(self, messages)
        self._lookahead_runtime.reset()

    async def aset(self, messages: list[ChatMessage]) -> None:
        """Set chat history asynchronously."""
        await asyncio.to_thread(ChatSummaryMemoryBuffer.set, self, messages)
        self._lookahead_runtime.reset()

    def reset(self) -> None:
        """Reset chat history."""
        ChatSummaryMemoryBuffer.reset(self)
        self._lookahead_runtime.reset()

    async def areset(self) -> None:
        """Reset chat history asynchronously."""
        await asyncio.to_thread(ChatSummaryMemoryBuffer.reset, self)
        self._lookahead_runtime.reset()

    def _replace_history(self, messages: list[ChatMessage]) -> None:
        ChatSummaryMemoryBuffer.reset(self)
        ChatSummaryMemoryBuffer.set(self, messages)

    def _configure_lookahead(self) -> None:
        config = SummarizationLookaheadConfig(
            token_limit=self.token_limit,
            soft_token_limit=self.soft_token_limit,
            hard_token_limit=self.hard_token_limit,
            recent_token_budget=self.recent_token_budget,
            count_initial_tokens=self.count_initial_tokens,
            summarize_prompt=self.summarize_prompt or SUMMARIZE_PROMPT,
            summary_extra_body=self.summary_extra_body,
            summary_llm_kwargs=self.summary_llm_kwargs,
            mark_sglang_lookahead=self.mark_sglang_lookahead,
        )
        self._lookahead_strategy = SummarizationLookaheadStrategy(
            self.llm,
            config=config,
            tokenizer_fn=self.tokenizer_fn,
        )
        self._lookahead_runtime = LookaheadRuntime(
            self._lookahead_strategy, agent_id=self.lookahead_agent_id
        )
