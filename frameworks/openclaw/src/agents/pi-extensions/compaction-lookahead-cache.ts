const LOOKAHEAD_CACHE_TTL_MS = 30 * 60 * 1000;

export type CompactionLookaheadStatus = "in_flight" | "ready";

export type CompactionLookaheadCandidate = {
  sessionId: string;
  status: CompactionLookaheadStatus;
  createdAt: number;
  generation: number;
  summary?: string;
  firstKeptEntryId?: string;
  tokensBefore?: number;
  details?: unknown;
};

const readyCache = new Map<string, CompactionLookaheadCandidate>();
const pendingCache = new Map<string, CompactionLookaheadCandidate>();

function isExpired(candidate: CompactionLookaheadCandidate, now = Date.now()): boolean {
  return now - candidate.createdAt > LOOKAHEAD_CACHE_TTL_MS;
}

function getFreshCandidate(
  cache: Map<string, CompactionLookaheadCandidate>,
  sessionId: string,
): CompactionLookaheadCandidate | undefined {
  const candidate = cache.get(sessionId);
  if (!candidate) {
    return undefined;
  }
  if (isExpired(candidate)) {
    cache.delete(sessionId);
    return undefined;
  }
  return candidate;
}

export function getCompactionLookaheadCandidate(
  sessionId: string,
): CompactionLookaheadCandidate | undefined {
  return getFreshCandidate(readyCache, sessionId) ?? getFreshCandidate(pendingCache, sessionId);
}

export function getCompactionLookaheadPendingCandidate(
  sessionId: string,
): CompactionLookaheadCandidate | undefined {
  return getFreshCandidate(pendingCache, sessionId);
}

export function getCompactionLookaheadReadyCandidate(
  sessionId: string,
): CompactionLookaheadCandidate | undefined {
  return getFreshCandidate(readyCache, sessionId);
}

export function beginCompactionLookahead(sessionId: string): CompactionLookaheadCandidate {
  const previousPending = getFreshCandidate(pendingCache, sessionId);
  const previousReady = getFreshCandidate(readyCache, sessionId);
  const previousGeneration = Math.max(
    previousPending?.generation ?? 0,
    previousReady?.generation ?? 0,
  );
  const candidate: CompactionLookaheadCandidate = {
    sessionId,
    status: "in_flight",
    createdAt: Date.now(),
    generation: previousGeneration + 1,
  };
  pendingCache.set(sessionId, candidate);
  return candidate;
}

export function resolveCompactionLookahead(params: {
  sessionId: string;
  generation: number;
  summary: string;
  firstKeptEntryId?: string;
  tokensBefore?: number;
  details?: unknown;
}): CompactionLookaheadCandidate | undefined {
  const current = getFreshCandidate(pendingCache, params.sessionId);
  if (!current || current.generation !== params.generation) {
    return undefined;
  }
  const resolved: CompactionLookaheadCandidate = {
    sessionId: params.sessionId,
    status: "ready",
    createdAt: Date.now(),
    generation: params.generation,
    summary: params.summary,
    firstKeptEntryId: params.firstKeptEntryId,
    tokensBefore: params.tokensBefore,
    details: params.details,
  };
  pendingCache.delete(params.sessionId);
  readyCache.set(params.sessionId, resolved);
  return resolved;
}

export function failCompactionLookahead(params: { sessionId: string; generation: number }): void {
  const current = getFreshCandidate(pendingCache, params.sessionId);
  if (current && current.generation === params.generation) {
    pendingCache.delete(params.sessionId);
  }
}

export function clearCompactionLookahead(sessionId: string): void {
  pendingCache.delete(sessionId);
  readyCache.delete(sessionId);
}

export const __testing = {
  LOOKAHEAD_CACHE_TTL_MS,
  pendingCache,
  readyCache,
};
