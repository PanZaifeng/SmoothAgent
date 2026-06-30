"""Core MiniAgent workflow and transform logic used by SmoothAgent."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

StrategyName = Literal["offloading", "sliding_window", "summarize", "sub_agent"]

STRATEGIES: tuple[StrategyName, ...] = (
    "offloading",
    "sliding_window",
    "summarize",
    "sub_agent",
)

WORD_POOL = (
    "agent context cache radix prefill decode scheduler lookahead commit "
    "summary offload sliding window subagent request response tool output "
    "module class function import branch exception test patch analysis trace "
    "latency throughput token batch queue stream generate prompt system user "
    "python repository caller callee symbol dependency file buffer result "
    "threshold trigger prepare consume artifact metadata decision observation"
).split()


@dataclass(slots=True)
class Message:
    role: str
    content: str


@dataclass(slots=True)
class StrategyConfig:
    """Configuration for MiniAgent workflow projections and transforms."""

    agent_id: int = 0
    seed: int = 42
    assistant_words: int = 400
    tool_words: int = 150
    long_tool_words: int = 2200
    sliding_keep_tokens: int = 10_000
    sliding_keep_turns: int = 0
    summarize_keep_recent_tokens: int = 4_000
    summary_max_tokens: int = 128
    offload_token_limit: int = 50
    sub_agent_system_words: int = 1600
    sub_agent_context_tokens: int = 7000
    sub_agent_instruction_words: int = 0


@dataclass(slots=True)
class TransformResult:
    messages: list[Message]
    metadata: dict[str, Any]


def words(seed: str, n: int) -> str:
    out: list[str] = []
    state = hashlib.sha256(seed.encode()).digest()
    for i in range(n):
        b = state[i % len(state)]
        out.append(WORD_POOL[(b + i * 17) % len(WORD_POOL)])
        if i % len(state) == len(state) - 1:
            state = hashlib.sha256(state + seed.encode()).digest()
    return " ".join(out)


def approximate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def serialize_messages(messages: list[Message]) -> str:
    labels = {
        "system": "System",
        "user": "Human",
        "assistant": "AI",
        "tool": "Tool",
    }
    return "\n".join(f"{labels.get(m.role, m.role.title())}: {m.content}" for m in messages)


def make_initial_messages(agent_id: int, seed: int) -> list[Message]:
    unique = hashlib.sha256(f"{agent_id}|{seed}".encode()).hexdigest()[:12]
    return [
        Message(
            "system",
            "You are a senior code analysis agent. "
            f"[AGENT-{agent_id}-{unique}] Keep conclusions precise.",
        )
    ]


def step_user_message(agent_id: int, step: int, seed: int) -> Message:
    body = words(f"user|{agent_id}|{step}|{seed}", 22)
    return Message("user", f"Step {step}: inspect the next code excerpt. {body}")


def assistant_message(agent_id: int, step: int, cfg: StrategyConfig) -> Message:
    body = words(f"assistant|{agent_id}|{step}|{cfg.seed}", cfg.assistant_words)
    return Message("assistant", f"Analysis for step {step}: {body}")


def tool_message(agent_id: int, step: int, cfg: StrategyConfig) -> Message:
    word_count = cfg.long_tool_words if step in (9, 18, 27) else cfg.tool_words
    body = words(f"tool|{agent_id}|{step}|{cfg.seed}", word_count)
    command = f"sed -n '{step},{step + 60}p' src/module_{step % 7}.py"
    return Message("tool", f"$ {command}\n{body}")


def keep_recent_by_tokens(messages: list[Message], token_budget: int) -> list[Message]:
    system = [m for m in messages if m.role == "system"][:1]
    non_system = [m for m in messages if m.role != "system"]
    kept: list[Message] = []
    total = approximate_token_count(serialize_messages(system))
    for msg in reversed(non_system):
        candidate = [msg] + kept
        if total + approximate_token_count(serialize_messages(candidate)) > token_budget and kept:
            break
        kept = candidate
    return system + kept


def keep_recent_by_turns(messages: list[Message], turns: int) -> list[Message]:
    if turns <= 0:
        return list(messages)
    system = [m for m in messages if m.role == "system"][:1]
    non_system = [m for m in messages if m.role != "system"]
    user_count = 0
    start = 0
    for idx in range(len(non_system) - 1, -1, -1):
        if non_system[idx].role == "user":
            user_count += 1
            if user_count == turns:
                start = idx
                break
    return system + non_system[start:]


def offload_messages(messages: list[Message], token_limit: int) -> list[Message]:
    out: list[Message] = []
    for msg in messages:
        if msg.role == "tool" and approximate_token_count(msg.content) > token_limit:
            digest = hashlib.sha256(msg.content.encode()).hexdigest()[:12]
            out.append(Message("tool", f"[Offloaded tool output: {digest}]"))
        else:
            out.append(msg)
    return out


def summary_split(
    messages: list[Message],
    keep_recent_tokens: int,
) -> tuple[list[Message], list[Message]]:
    system = [m for m in messages if m.role == "system"][:1]
    non_system = [m for m in messages if m.role != "system"]
    tail: list[Message] = []
    for msg in reversed(non_system):
        candidate = [msg] + tail
        if approximate_token_count(serialize_messages(candidate)) > keep_recent_tokens and tail:
            break
        tail = candidate
    source_count = max(1, len(non_system) - len(tail))
    source = system + non_system[:source_count]
    tail_msgs = non_system[source_count:] or non_system[-1:]
    return source, tail_msgs


def summary_messages(summary: str, tail_msgs: list[Message]) -> list[Message]:
    return [Message("system", summary)] + list(tail_msgs)


def sub_agent_messages(messages: list[Message], cfg: StrategyConfig) -> list[Message]:
    system = (
        "You are a security-audit sub-agent. "
        + words(f"sub_system|{cfg.agent_id}|{cfg.seed}", cfg.sub_agent_system_words)
    )
    recent = keep_recent_by_tokens(messages, cfg.sub_agent_context_tokens)
    delegation = words(
        f"sub_instruction|{cfg.agent_id}|{cfg.seed}",
        cfg.sub_agent_instruction_words,
    )
    instruction = (
        "Delegate task: review the following context and continue the code analysis.\n"
        + delegation
        + "\n"
        + serialize_messages(recent)
    )
    return [Message("system", system), Message("user", instruction)]


def transform_messages(
    strategy: StrategyName,
    messages: list[Message],
    cfg: StrategyConfig,
    summary_text: str = "",
) -> TransformResult:
    if strategy == "offloading":
        transformed = offload_messages(messages, cfg.offload_token_limit)
        return TransformResult(transformed, {"offload_token_limit": cfg.offload_token_limit})
    if strategy == "sliding_window":
        if cfg.sliding_keep_turns > 0:
            transformed = keep_recent_by_turns(messages, cfg.sliding_keep_turns)
        else:
            transformed = keep_recent_by_tokens(messages, cfg.sliding_keep_tokens)
        return TransformResult(transformed, {})
    if strategy == "summarize":
        source, tail = summary_split(messages, cfg.summarize_keep_recent_tokens)
        summary = summary_text or (
            "Context summary:\n"
            + " ".join(serialize_messages(source).split()[: cfg.summary_max_tokens])
        )
        return TransformResult(
            summary_messages(summary, tail),
            {
                "source_text": serialize_messages(source),
                "tail_text": serialize_messages(tail),
                "replace_count": len(source),
            },
        )
    if strategy == "sub_agent":
        transformed = sub_agent_messages(messages, cfg)
        return TransformResult(
            transformed,
            {"sub_agent_system_prompt": transformed[0].content},
        )
    raise ValueError(f"unknown strategy: {strategy}")


def project_messages_to_step(
    messages: list[Message],
    cfg: StrategyConfig,
    *,
    current_step: int,
    target_step: int,
    current_step_completed: bool,
) -> list[Message]:
    if target_step < current_step:
        return list(messages)
    projected = list(messages)
    if target_step == current_step:
        return projected

    if not current_step_completed:
        projected.append(assistant_message(cfg.agent_id, current_step, cfg))
        projected.append(tool_message(cfg.agent_id, current_step, cfg))

    for step in range(current_step + 1, target_step):
        projected.append(step_user_message(cfg.agent_id, step, cfg.seed))
        projected.append(assistant_message(cfg.agent_id, step, cfg))
        projected.append(tool_message(cfg.agent_id, step, cfg))
    projected.append(step_user_message(cfg.agent_id, target_step, cfg.seed))
    return projected


def should_lookahead(
    *,
    step: int,
    next_commit_step: int | None,
    strategy: StrategyName,
    warmup_lead_steps: dict[str, int],
    messages: list[Message] | None = None,
    summarize_soft_trigger_tokens: int = 0,
) -> bool:
    if next_commit_step is None or step >= next_commit_step:
        return False
    if (
        strategy == "summarize"
        and messages is not None
        and summarize_soft_trigger_tokens > 0
        and approximate_token_count(serialize_messages(messages)) >= summarize_soft_trigger_tokens
    ):
        return True
    lead = warmup_lead_steps.get(strategy, 0)
    return step >= max(1, next_commit_step - lead)


def build_summarize_generate_text(summary_text: str, tail_text: str) -> str:
    """Construct the summary+tail prompt shape used by the SGLang commit path."""

    return f"System: {summary_text}\n{tail_text}"
