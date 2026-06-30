# SmoothAgent

SmoothAgent is a lookahead context-engineering system for agent applications.
It moves deterministic context transformations off the foreground request path:
old context can be compacted, summarized, or windowed in the background, then
committed when the agent reaches a hard context boundary.

The repository contains the core SmoothAgent implementation for MiniAgent and
other framework integrations, plus SGLang scheduling support.

## Repository Layout

```text
backend/
  sglang/                 # SmoothAgent scheduling support for SGLang
miniagent/                # MiniAgent framework workflow and strategies
frameworks/
  langchain/              # LangChain SmoothAgent runtime and strategies
  llama_index/            # LlamaIndex memory-level lookahead integration
  autogen/                # AutoGen model-context lookahead engine
  openclaw/               # OpenClaw summarization lookahead integration
```

The framework directories preserve package-relative paths so the integrations
can be inspected or applied in-place.  `miniagent/` is kept as a top-level
importable MiniAgent package. It contains framework-facing components for
code-analysis workflows with interleaved LLM calls, tool observations, context
transformations, and foreground/background requests through an
OpenAI-compatible SGLang endpoint.

## Lookahead Interface

The shared strategy/runtime contract is:

- `MainState(messages, strategy_data)`
- `LookaheadState(transformed, last_segment_end, strategy_data)`
- `LookaheadStrategy.should_lookahead()`
- `LookaheadStrategy.transform()`
- `LookaheadStrategy.should_commit()`
- `LookaheadStrategy.should_promote()`
- `LookaheadRuntime.on_commit(messages)`
- `LookaheadRuntime.on_segment_boundary(messages)`

`transform()` is segment-incremental: it updates the transformed prefix from
the current segment and existing lookahead state, without relying on future
tokens.

This interface corresponds to the paper's lookahead programming model
(`sec:lookahead`, `subsec:programming_model`) and the strategy interface in
`list:lookahead`. Agent integrations call the commit and segment-boundary
hooks in the pattern shown by `list:agent_loop`.

## Backend

SmoothAgent uses SGLang through its OpenAI-compatible endpoint.  Foreground
requests and lookahead/background requests carry top-level request metadata
such as request class, priority class, lookahead group id, and lookahead flag.

The relevant SGLang code is under:

```text
backend/sglang/python/sglang/srt/managers/
```

That directory contains the SmoothAgent scheduler and request-metadata support
for SGLang.  To launch a serving backend, use a complete SGLang checkout that
includes these files.

The scheduler comments use the paper labels `eq:latency-model`,
`alg:schedule-prefill`, and `alg:schedule-hybrid` for the batch-latency model
and the PD-disaggregated / PD-co-located scheduling policies.

## Framework Coverage

| Framework | Strategy |
| --- | --- |
| MiniAgent | offloading, sliding window, summarization, sub-agent |
| LangChain | sliding window, summarization |
| LlamaIndex | sliding window, summarization |
| AutoGen | sliding window |
| OpenClaw | summarization |

Each framework integration keeps the framework-native hook point while adding
the SmoothAgent runtime state, strategy trigger/commit logic, stale-result
checks, and SGLang foreground/background request metadata.

## Request Metadata

SmoothAgent uses top-level OpenAI-compatible request fields:

- `request_class`
- `is_lookahead`
- `priority_class`
- `lookahead_group_id`

Foreground requests use latency-critical metadata.  Background lookahead
requests use best-effort metadata and the same SGLang endpoint.
