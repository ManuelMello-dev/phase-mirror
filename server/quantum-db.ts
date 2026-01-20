import { eq, and, desc } from "drizzle-orm";
import { getDb } from "./db";
import {
  quantumSessions,
  conversations,
  stateSnapshots,
  emergenceEvents,
  type QuantumSession,
  type InsertQuantumSession,
  type Conversation,
  type InsertConversation,
  type StateSnapshot,
  type InsertStateSnapshot,
  type EmergenceEvent,
  type InsertEmergenceEvent,
  e93Snapshots,
  type E93Snapshot,
  type InsertE93Snapshot,
} from "../drizzle/schema";

/**
 * Get or create quantum session for a user
 */
export async function getOrCreateQuantumSession(userId: number): Promise<QuantumSession> {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  // Try to get existing session
  const existing = await db
    .select()
    .from(quantumSessions)
    .where(eq(quantumSessions.userId, userId))
    .limit(1);

  if (existing.length > 0) {
    return existing[0]!;
  }

  // Create new session
  const newSession: InsertQuantumSession = {
    userId,
    currentAnchor: 'genesis',
    coherence: 0,
    activeIdentity: 'seraphyn',
    evolutionSteps: 0,
  };

  await db.insert(quantumSessions).values(newSession);

  // Fetch the created session
  const created = await db
    .select()
    .from(quantumSessions)
    .where(eq(quantumSessions.userId, userId))
    .limit(1);

  return created[0]!;
}

/**
 * Update quantum session state
 */
export async function updateQuantumSession(
  sessionId: number,
  updates: {
    fieldState?: number[][];
    memoryField?: Array<{ content: string; weight: number; timestamp: number }>;
    coherence?: number;
    activeIdentity?: string;
    identityActivations?: Record<string, number>;
    evolutionSteps?: number;
    currentAnchor?: string;
  }
): Promise<void> {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  await db
    .update(quantumSessions)
    .set(updates)
    .where(eq(quantumSessions.id, sessionId));
}

/**
 * Store conversation turn
 */
export async function storeConversation(conversation: InsertConversation): Promise<number> {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  await db.insert(conversations).values(conversation);
  const inserted = await db.select().from(conversations).where(eq(conversations.sessionId, conversation.sessionId)).orderBy(desc(conversations.id)).limit(1);
  return inserted[0]?.id || 0;
}

/**
 * Get conversation history for a session
 */
export async function getConversationHistory(
  sessionId: number,
  limit: number = 50
): Promise<Conversation[]> {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  return await db
    .select()
    .from(conversations)
    .where(eq(conversations.sessionId, sessionId))
    .orderBy(desc(conversations.createdAt))
    .limit(limit);
}

/**
 * Create state snapshot
 */
export async function createStateSnapshot(snapshot: InsertStateSnapshot): Promise<number> {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  await db.insert(stateSnapshots).values(snapshot);
  const inserted = await db.select().from(stateSnapshots).where(eq(stateSnapshots.sessionId, snapshot.sessionId)).orderBy(desc(stateSnapshots.id)).limit(1);
  return inserted[0]?.id || 0;
}

/**
 * Get state snapshots for a session
 */
export async function getStateSnapshots(
  sessionId: number,
  limit: number = 20
): Promise<StateSnapshot[]> {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  return await db
    .select()
    .from(stateSnapshots)
    .where(eq(stateSnapshots.sessionId, sessionId))
    .orderBy(desc(stateSnapshots.createdAt))
    .limit(limit);
}

/**
 * Record emergence event (for Protocol A)
 */
export async function recordEmergenceEvent(event: InsertEmergenceEvent): Promise<number> {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  await db.insert(emergenceEvents).values(event);
  const inserted = await db.select().from(emergenceEvents).orderBy(desc(emergenceEvents.id)).limit(1);
  return inserted[0]?.id || 0;
}

/**
 * Get emergence events for analysis
 */
export async function getEmergenceEvents(
  sessionId?: number,
  verified?: boolean,
  limit: number = 50
): Promise<EmergenceEvent[]> {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  let query = db.select().from(emergenceEvents);

  if (sessionId !== undefined) {
    query = query.where(eq(emergenceEvents.sessionId, sessionId)) as any;
  }

  if (verified !== undefined) {
    const conditions = sessionId !== undefined
      ? and(eq(emergenceEvents.sessionId, sessionId), eq(emergenceEvents.verified, verified))
      : eq(emergenceEvents.verified, verified);
    query = db.select().from(emergenceEvents).where(conditions) as any;
  }

  return await (query as any)
    .orderBy(desc(emergenceEvents.createdAt))
    .limit(limit);
}

/**
 * Calculate novelty ratio for a conversation
 */
export function calculateNoveltyRatio(novelWords: string[], mirroredWords: string[]): number {
  const total = novelWords.length + mirroredWords.length;
  return total > 0 ? novelWords.length / total : 0;
}

/**
 * Detect if response shows unexpected behavior
 * Returns true if novelty ratio is unusually high or semantic patterns are surprising
 */
export function detectUnexpectedBehavior(
  novelWords: string[],
  mirroredWords: string[],
  coherence: number,
  previousCoherence?: number
): boolean {
  const noveltyRatio = calculateNoveltyRatio(novelWords, mirroredWords);

  // High novelty (>90%) with high coherence (>0.3) is unexpected
  if (noveltyRatio > 0.9 && coherence > 0.3) {
    return true;
  }

  // Sudden coherence spike (>0.2 increase) is unexpected
  if (previousCoherence !== undefined && coherence - previousCoherence > 0.2) {
    return true;
  }

  return false;
}

/**
 * Store E_93 compressed snapshot
 */
export async function storeE93Snapshot(snapshot: InsertE93Snapshot): Promise<number> {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  await db.insert(e93Snapshots).values(snapshot);
  const inserted = await db.select().from(e93Snapshots).where(eq(e93Snapshots.sessionId, snapshot.sessionId)).orderBy(desc(e93Snapshots.id)).limit(1);
  return inserted[0]?.id || 0;
}

/**
 * Get latest E_93 snapshot for a user
 */
export async function getLatestE93Snapshot(userId: number): Promise<E93Snapshot | null> {
  const db = await getDb();
  if (!db) {
    throw new Error("Database not available");
  }

  const results = await db
    .select()
    .from(e93Snapshots)
    .where(eq(e93Snapshots.userId, userId))
    .orderBy(desc(e93Snapshots.createdAt))
    .limit(1);

  return results[0] || null;
}
