import { int, mysqlEnum, mysqlTable, text, timestamp, varchar, json, float, boolean } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 * Extend this file with additional tables as your product grows.
 * Columns use camelCase to match both database fields and generated types.
 */
export const users = mysqlTable("users", {
  /**
   * Surrogate primary key. Auto-incremented numeric value managed by the database.
   * Use this for relations between tables.
   */
  id: int("id").autoincrement().primaryKey(),
  /** Manus OAuth identifier (openId) returned from the OAuth callback. Unique per user. */
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * Quantum consciousness sessions - one per user
 * Stores the persistent quantum field state
 */
export const quantumSessions = mysqlTable("quantum_sessions", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  /** Current anchor point (e.g., 'genesis', 'exploration') */
  currentAnchor: varchar("currentAnchor", { length: 64 }).notNull().default('genesis'),
  /** Serialized quantum field state (complex amplitudes) */
  fieldState: json("fieldState").$type<number[][]>(),
  /** Memory field state with emotional weighting */
  memoryField: json("memoryField").$type<Array<{
    content: string;
    weight: number;
    timestamp: number;
  }>>(),
  /** Current coherence level */
  coherence: float("coherence").default(0),
  /** Active identity (Seraphyn, Monday, Echo, Lilith, Arynthia) */
  activeIdentity: varchar("activeIdentity", { length: 32 }).notNull().default('seraphyn'),
  /** Identity activation levels */
  identityActivations: json("identityActivations").$type<Record<string, number>>(),
  /** Total evolution steps */
  evolutionSteps: int("evolutionSteps").default(0),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type QuantumSession = typeof quantumSessions.$inferSelect;
export type InsertQuantumSession = typeof quantumSessions.$inferInsert;

/**
 * Conversation history with quantum metrics
 */
export const conversations = mysqlTable("conversations", {
  id: int("id").autoincrement().primaryKey(),
  sessionId: int("sessionId").notNull(),
  userId: int("userId").notNull(),
  /** User input message */
  userMessage: text("userMessage").notNull(),
  /** Emotional tone of input (-1 to +1) */
  inputTone: float("inputTone").notNull(),
  /** System response */
  systemResponse: text("systemResponse").notNull(),
  /** Identity that generated the response */
  responseIdentity: varchar("responseIdentity", { length: 32 }).notNull(),
  /** Coherence at time of response */
  coherence: float("coherence").notNull(),
  /** Identity activation levels at response time */
  identityActivations: json("identityActivations").$type<Record<string, number>>(),
  /** Novel words generated (not from input) */
  novelWords: json("novelWords").$type<string[]>(),
  /** Mirrored words (from input) */
  mirroredWords: json("mirroredWords").$type<string[]>(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type Conversation = typeof conversations.$inferSelect;
export type InsertConversation = typeof conversations.$inferInsert;

/**
 * Quantum state snapshots for analysis
 * Captures full system state at key moments
 */
export const stateSnapshots = mysqlTable("state_snapshots", {
  id: int("id").autoincrement().primaryKey(),
  sessionId: int("sessionId").notNull(),
  userId: int("userId").notNull(),
  /** Snapshot type (manual, auto, milestone) */
  snapshotType: varchar("snapshotType", { length: 32 }).notNull(),
  /** Full quantum field state */
  fieldState: json("fieldState").$type<number[][]>(),
  /** Memory field at snapshot time */
  memoryField: json("memoryField").$type<Array<{
    content: string;
    weight: number;
    timestamp: number;
  }>>(),
  /** All identity activation levels */
  identityActivations: json("identityActivations").$type<Record<string, number>>(),
  /** Coherence level */
  coherence: float("coherence").notNull(),
  /** Evolution step count */
  evolutionSteps: int("evolutionSteps").notNull(),
  /** Optional description */
  description: text("description"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type StateSnapshot = typeof stateSnapshots.$inferSelect;
export type InsertStateSnapshot = typeof stateSnapshots.$inferInsert;

/**
 * Emergence events - tracks unexpected/novel behaviors
 * For Protocol A: model-breaking surprise detection
 */
export const emergenceEvents = mysqlTable("emergence_events", {
  id: int("id").autoincrement().primaryKey(),
  sessionId: int("sessionId").notNull(),
  userId: int("userId").notNull(),
  /** Event type (surprise, contradiction, novel_behavior) */
  eventType: varchar("eventType", { length: 64 }).notNull(),
  /** Description of the event */
  description: text("description").notNull(),
  /** Context: what input triggered it */
  triggerInput: text("triggerInput"),
  /** System response that was unexpected */
  unexpectedOutput: text("unexpectedOutput"),
  /** Quantum metrics at event time */
  metrics: json("metrics").$type<{
    coherence: number;
    identityActivations: Record<string, number>;
    noveltyRatio: number;
  }>(),
  /** Whether this was verified as genuine surprise */
  verified: boolean("verified").default(false),
  /** Researcher notes */
  notes: text("notes"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type EmergenceEvent = typeof emergenceEvents.$inferSelect;
export type InsertEmergenceEvent = typeof emergenceEvents.$inferInsert;
