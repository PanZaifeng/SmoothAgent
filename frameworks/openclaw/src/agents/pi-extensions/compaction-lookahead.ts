import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { OpenClawConfig } from "../../config/config.js";
import { createSubsystemLogger } from "../../logging/subsystem.js";
import { estimateMessagesTokens } from "../compaction.js";
import type { CompactEmbeddedPiSessionParams } from "../pi-embedded-runner/compact.js";
import { computeCompactionSafeguardRuntimeValue } from "../pi-embedded-runner/extensions.js";
import { prepareEmbeddedPiSessionCompaction } from "../pi-embedded-runner/compact.runtime.js";
import {
  beginCompactionLookahead,
  failCompactionLookahead,
  getCompactionLookaheadPendingCandidate,
  resolveCompactionLookahead,
} from "./compaction-lookahead-cache.js";
import {
  buildCompactionSummaryFromPreparation,
  createSglangRequestClassExtraOptions,
  type CompactionPreparationLike,
} from "./compaction-safeguard.js";

const log = createSubsystemLogger("compaction-lookahead");
const DEFAULT_SOFT_TRIGGER_RATIO = 0.6;

function buildCompactionLookaheadGroupId(params: {
  sessionId: string;
  generation: number;
}): string {
  return `openclaw-compaction-lookahead:${params.sessionId}:${params.generation}`;
}

function logLookaheadEvent(params: {
  event: "lookahead_start" | "lookahead_ready" | "lookahead_failed";
  sessionId: string;
  provider?: string;
  model?: string;
  durationMs?: number;
  tokensBefore?: number;
  usedCachedSummary: boolean;
  reason?: string;
}): void {
  log.info(params.event, {
    event: params.event,
    sessionId: params.sessionId,
    provider: params.provider,
    model: params.model,
    trigger: "soft",
    durationMs: params.durationMs,
    tokensBefore: params.tokensBefore,
    usedCachedSummary: params.usedCachedSummary,
    reason: params.reason,
  });
}

function resolveSoftTriggerUsage(params: {
  messages: AgentMessage[];
  tokenBudget?: number;
  currentTokenCount?: number;
}): number | undefined {
  if (!params.tokenBudget || params.tokenBudget <= 0) {
    return undefined;
  }
  if (
    typeof params.currentTokenCount === "number" &&
    Number.isFinite(params.currentTokenCount) &&
    params.currentTokenCount > 0
  ) {
    return params.currentTokenCount / params.tokenBudget;
  }
  if (params.messages.length === 0) {
    return undefined;
  }
  try {
    return estimateMessagesTokens(params.messages) / params.tokenBudget;
  } catch {
    return undefined;
  }
}

function resolveSoftTriggerRatio(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value) && value > 0 && value <= 1) {
    return value;
  }
  return DEFAULT_SOFT_TRIGGER_RATIO;
}

export function scheduleCompactionLookahead(params: {
  sessionId: string;
  sessionFile: string;
  messages: AgentMessage[];
  tokenBudget?: number;
  runtimeContext?: Record<string, unknown>;
}): void {
  const runtimeContext = (params.runtimeContext ?? {}) as Partial<CompactEmbeddedPiSessionParams>;
  const compactionCfg = runtimeContext.config?.agents?.defaults?.compaction;
  const compactionMode = compactionCfg?.mode;
  if (compactionMode !== "safeguard") {
    return;
  }

  const softTriggerRatio = resolveSoftTriggerRatio(compactionCfg?.softTriggerRatio);
  const usage = resolveSoftTriggerUsage({
    messages: params.messages,
    tokenBudget:
      params.tokenBudget ??
      (typeof runtimeContext.tokenBudget === "number" ? runtimeContext.tokenBudget : undefined),
    currentTokenCount:
      typeof runtimeContext.currentTokenCount === "number"
        ? runtimeContext.currentTokenCount
        : undefined,
  });
  if (usage === undefined || usage < softTriggerRatio) {
    return;
  }

  const existingCandidate = getCompactionLookaheadPendingCandidate(params.sessionId);
  if (existingCandidate?.status === "in_flight") {
    return;
  }

  const candidate = beginCompactionLookahead(params.sessionId);
  logLookaheadEvent({
    event: "lookahead_start",
    sessionId: params.sessionId,
    provider: runtimeContext.provider,
    model: runtimeContext.model,
    usedCachedSummary: false,
  });

  void runCompactionLookahead({
    sessionId: params.sessionId,
    sessionFile: params.sessionFile,
    runtimeContext,
    generation: candidate.generation,
  });
}

async function runCompactionLookahead(params: {
  sessionId: string;
  sessionFile: string;
  generation: number;
  runtimeContext: Partial<CompactEmbeddedPiSessionParams>;
}): Promise<void> {
  const startedAt = Date.now();
  try {
    const prepared = await prepareEmbeddedPiSessionCompaction({
      ...params.runtimeContext,
      sessionId: params.sessionId,
      sessionFile: params.sessionFile,
      workspaceDir: params.runtimeContext.workspaceDir ?? process.cwd(),
    } as CompactEmbeddedPiSessionParams);
    if (!prepared.ok) {
      failCompactionLookahead({
        sessionId: params.sessionId,
        generation: params.generation,
      });
      logLookaheadEvent({
        event: "lookahead_failed",
        sessionId: params.sessionId,
        provider: params.runtimeContext.provider,
        model: params.runtimeContext.model,
        durationMs: Date.now() - startedAt,
        usedCachedSummary: false,
        reason: prepared.reason,
      });
      return;
    }

    const runtime = computeCompactionSafeguardRuntimeValue({
      cfg: params.runtimeContext.config as OpenClawConfig | undefined,
      provider: prepared.provider,
      modelId: prepared.modelId,
      model: prepared.model,
    });
    const result = await buildCompactionSummaryFromPreparation({
      preparation: prepared.preparation as unknown as CompactionPreparationLike,
      signal: new AbortController().signal,
      model: prepared.model,
      apiKey: prepared.apiKey,
      runtime,
      workspaceDir: prepared.workspaceDir,
      sessionId: params.sessionId,
      trigger: "soft",
      extraOptions:
        runtime?.sglangSchedulingEnabled && prepared.model.api === "openai-completions"
          ? createSglangRequestClassExtraOptions("bg", {
              lookaheadGroupId: buildCompactionLookaheadGroupId({
                sessionId: params.sessionId,
                generation: params.generation,
              }),
            })
          : undefined,
    });
    const resolved = resolveCompactionLookahead({
      sessionId: params.sessionId,
      generation: params.generation,
      summary: result.summary,
      firstKeptEntryId: result.firstKeptEntryId,
      tokensBefore: result.tokensBefore,
      details: result.details,
    });
    if (!resolved) {
      return;
    }
    logLookaheadEvent({
      event: "lookahead_ready",
      sessionId: params.sessionId,
      provider: prepared.provider,
      model: prepared.modelId,
      durationMs: Date.now() - startedAt,
      tokensBefore: result.tokensBefore,
      usedCachedSummary: false,
    });
  } catch (error) {
    failCompactionLookahead({
      sessionId: params.sessionId,
      generation: params.generation,
    });
    logLookaheadEvent({
      event: "lookahead_failed",
      sessionId: params.sessionId,
      provider: params.runtimeContext.provider,
      model: params.runtimeContext.model,
      durationMs: Date.now() - startedAt,
      usedCachedSummary: false,
      reason: error instanceof Error ? error.message : String(error),
    });
  }
}

export const __testing = {
  DEFAULT_SOFT_TRIGGER_RATIO,
  buildCompactionLookaheadGroupId,
  resolveSoftTriggerRatio,
  resolveSoftTriggerUsage,
};
