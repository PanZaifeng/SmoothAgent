"""SmoothAgent lookahead context-engineering runtime.

This module implements the agent-side abstractions described in the
SmoothAgent paper and runtime design:

- :class:`MainState` / :class:`LookaheadState` data classes.
- :class:`LookaheadStrategy` abstract base class with the four-method
  contract (``should_lookahead`` / ``transform`` / ``should_commit`` /
  ``should_promote``).
- Three reference strategies: offloading, sliding-window / keep-recent-K,
  and summarization.
- :class:`LookaheadRuntime` coordinating the main and lookahead streams.
- :class:`LookaheadRequestMeta` and :func:`build_extra_body` helpers used
  to mark an outbound LLM request as a best-effort lookahead request when
  it is forwarded to an SGLang serving instance.

The SmoothAgent module is intentionally additive. The pre-existing
control-plane dispatcher in :mod:`langchain_classic.memory.lookahead` and
the summarization middleware in :mod:`langchain.agents.middleware` continue
to work unchanged.
"""

from langchain_classic.experimental.smoothagent.offload_store import (
    OffloadRecord,
    OffloadStore,
)
from langchain_classic.experimental.smoothagent.request_meta import (
    LookaheadRequestMeta,
    build_extra_body,
    is_be_request,
)
from langchain_classic.experimental.smoothagent.runtime import (
    LookaheadRuntime,
    message_signature,
)
from langchain_classic.experimental.smoothagent.sglang import (
    SGLangSummaryClient,
)
from langchain_classic.experimental.smoothagent.state import (
    LookaheadState,
    MainState,
)
from langchain_classic.experimental.smoothagent.strategies import (
    LookaheadStrategy,
    OffloadConfig,
    OffloadLookaheadStrategy,
    SlidingWindowConfig,
    SlidingWindowLookaheadStrategy,
    SummarizationConfig,
    SummarizationLookaheadStrategy,
)
from langchain_classic.experimental.smoothagent.token_utils import (
    count_message_tokens,
    count_tokens,
)

__all__ = [
    "LookaheadRequestMeta",
    "LookaheadRuntime",
    "LookaheadState",
    "LookaheadStrategy",
    "MainState",
    "OffloadConfig",
    "OffloadLookaheadStrategy",
    "OffloadRecord",
    "OffloadStore",
    "SlidingWindowConfig",
    "SlidingWindowLookaheadStrategy",
    "SGLangSummaryClient",
    "SummarizationConfig",
    "SummarizationLookaheadStrategy",
    "build_extra_body",
    "count_message_tokens",
    "count_tokens",
    "is_be_request",
    "message_signature",
]
