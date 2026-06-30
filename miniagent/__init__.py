"""MiniAgent framework components for SmoothAgent.

This package contains MiniAgent code-analysis workflows, context
transforms, SGLang OpenAI-compatible request metadata helpers, and TTFT row
building utilities.
"""

from miniagent.core import (
    Message,
    StrategyConfig,
    TransformResult,
    approximate_token_count,
    assistant_message,
    build_summarize_generate_text,
    make_initial_messages,
    project_messages_to_step,
    serialize_messages,
    should_lookahead,
    step_user_message,
    tool_message,
    transform_messages,
)
from miniagent.sglang import ChatResult, SmoothAgentRequestMeta, build_extra_body
from miniagent.workflow import (
    CSV_FIELDS,
    MiniAgentState,
    MiniAgentStep,
    append_workflow_turn,
    build_step_row,
    initial_state,
    request_meta,
    should_schedule_background,
    sync_commit_projection,
)

__all__ = [
    "ChatResult",
    "CSV_FIELDS",
    "Message",
    "MiniAgentState",
    "MiniAgentStep",
    "SmoothAgentRequestMeta",
    "StrategyConfig",
    "TransformResult",
    "append_workflow_turn",
    "approximate_token_count",
    "assistant_message",
    "build_extra_body",
    "build_summarize_generate_text",
    "build_step_row",
    "initial_state",
    "make_initial_messages",
    "project_messages_to_step",
    "request_meta",
    "serialize_messages",
    "should_schedule_background",
    "should_lookahead",
    "step_user_message",
    "sync_commit_projection",
    "tool_message",
    "transform_messages",
]
