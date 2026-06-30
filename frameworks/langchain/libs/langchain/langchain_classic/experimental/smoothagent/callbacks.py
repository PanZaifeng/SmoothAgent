"""Optional ``BaseCallbackHandler`` adapter for the lookahead runtime.

Use :class:`LookaheadCallbackHandler` to drive a
:class:`~langchain_classic.experimental.smoothagent.runtime.LookaheadRuntime`
from a standard langchain agent executor without touching the agent loop.

Note: ``BaseCallbackHandler`` does not expose a hook *before* the next LLM
call. The runtime's :meth:`on_commit` therefore needs to be wired into the
agent loop or middleware separately. The callback handler covers the
``on_segment_boundary`` half of the contract — it fires after every
agent action / observation.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from langchain_classic.experimental.smoothagent.runtime import LookaheadRuntime


class LookaheadCallbackHandler:
    """Lightweight adapter that calls ``on_segment_boundary`` after each step.

    The class is intentionally not a subclass of
    ``langchain_core.callbacks.BaseCallbackHandler`` to keep this module
    importable without pulling the full callback machinery — most agent
    executors accept any object with ``on_agent_action`` /
    ``on_tool_end`` / ``on_llm_end`` methods.
    """

    def __init__(
        self,
        runtime: LookaheadRuntime,
        *,
        message_supplier: "MessageSupplier",
    ) -> None:
        self.runtime = runtime
        self._message_supplier = message_supplier
        self._lock = threading.Lock()

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        self._dispatch_segment_boundary()

    def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        self._dispatch_segment_boundary()

    def on_agent_action(self, *args: Any, **kwargs: Any) -> None:
        self._dispatch_segment_boundary()

    def _dispatch_segment_boundary(self) -> None:
        with self._lock:
            messages = self._message_supplier()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        coro = self.runtime.on_segment_boundary(messages)
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            asyncio.run(coro)


from collections.abc import Callable  # noqa: E402

MessageSupplier = Callable[[], list[Any]]
"""Returns the live message list for the agent."""
