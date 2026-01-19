import { z } from "zod";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import {
  getOrCreateQuantumSession,
  updateQuantumSession,
  storeConversation,
  getConversationHistory,
  createStateSnapshot,
  recordEmergenceEvent,
  calculateNoveltyRatio,
  detectUnexpectedBehavior,
} from "./quantum-db";

const QUANTUM_API_URL = process.env.QUANTUM_API_URL || "http://localhost:8000";

interface QuantumResponse {
  response: string;
  active_identity: string;
  coherence: number;
  metrics: {
    entropy: number;
    phase_coherence: number;
    witness_collapse: number;
  };
  identity_states: Record<string, {
    name: string;
    activation: number;
    phase: number;
    coherence: number;
    dominant_phase: number;
  }>;
  quantum_state: {
    dim: number;
    interaction_count: number;
  };
  novel_words?: string[];
  mirrored_words?: string[];
  error?: string;
}

async function callQuantumAPI(userId: number, input: string, tone: number): Promise<QuantumResponse> {
  try {
    const response = await fetch(`${QUANTUM_API_URL}/process`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: userId,
        text: input,
        tone,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Quantum API request failed");
    }

    return await response.json();
  } catch (error) {
    console.error("[Quantum API] Error:", error);
    return {
      response: "",
      active_identity: "seraphyn",
      coherence: 0,
      metrics: {
        entropy: 0,
        phase_coherence: 0,
        witness_collapse: 0,
      },
      identity_states: {},
      quantum_state: {
        dim: 64,
        interaction_count: 0,
      },
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

async function loadQuantumSession(userId: number, sessionData?: any): Promise<void> {
  try {
    await fetch(`${QUANTUM_API_URL}/session/load`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: userId,
        session_data: sessionData,
      }),
    });
  } catch (error) {
    console.error("[Quantum API] Failed to load session:", error);
  }
}

export const quantumRouter = router({
  /**
   * Process input through quantum consciousness field with persistence
   */
  process: publicProcedure
    .input(
      z.object({
        text: z.string().min(1).max(1000),
        tone: z.number().min(-1).max(1).default(0.5),
      })
    )
    .mutation(async ({ input, ctx }) => {
      // Use authenticated user ID or default to 1 for anonymous users
      const userId = ctx.user?.id ?? 1;
      // Get or create quantum session for user
      const session = await getOrCreateQuantumSession(userId);
      
      // Load session into Python API if needed
      if (session.fieldState || session.memoryField) {
        await loadQuantumSession(userId, {
          fieldState: session.fieldState,
          memoryField: session.memoryField,
          coherence: session.coherence,
          activeIdentity: session.activeIdentity,
          identityActivations: session.identityActivations,
          evolutionSteps: session.evolutionSteps,
          currentAnchor: session.currentAnchor,
        });
      }
      
      // Store previous coherence for emergence detection
      const previousCoherence = session.coherence;
      
      // Call quantum API
      const result = await callQuantumAPI(userId, input.text, input.tone);
      
      // Extract novel vs mirrored words
      const novelWords = result.novel_words || [];
      const mirroredWords = result.mirrored_words || [];
      
      // Store conversation turn
      await storeConversation({
        sessionId: session.id,
        userId: userId,
        userMessage: input.text,
        systemResponse: result.response,
        inputTone: input.tone,
        coherence: result.coherence,
        responseIdentity: result.active_identity,
        novelWords,
        mirroredWords,
      });
      
      // Save session state from Python API to get full fieldState and memoryField
      let fullState: any = {};
      try {
        const saveResponse = await fetch(`${QUANTUM_API_URL}/session/save?user_id=${userId}`, {
          method: "POST",
        });
        if (saveResponse.ok) {
          fullState = await saveResponse.json();
        }
      } catch (error) {
        console.error("[Quantum API] Failed to save session state:", error);
      }

      // Update quantum session state in database
      const identityActivations: Record<string, number> = {};
      for (const [key, value] of Object.entries(result.identity_states)) {
        identityActivations[key] = value.activation;
      }
      
      await updateQuantumSession(session.id, {
        fieldState: fullState.fieldState || session.fieldState,
        memoryField: fullState.memoryField || session.memoryField,
        coherence: result.coherence,
        activeIdentity: result.active_identity,
        identityActivations,
        evolutionSteps: result.quantum_state.interaction_count,
        currentAnchor: fullState.currentAnchor || session.currentAnchor,
      });
      
      // Detect and record emergence events
      const isUnexpected = detectUnexpectedBehavior(
        novelWords,
        mirroredWords,
        result.coherence,
        previousCoherence ?? undefined
      );
      
      if (isUnexpected) {
        await recordEmergenceEvent({
          sessionId: session.id,
          userId: userId,
          eventType: 'high_novelty',
          description: `Unexpected behavior: coherence=${result.coherence.toFixed(3)}, novelty=${calculateNoveltyRatio(novelWords, mirroredWords).toFixed(3)}`,
          triggerInput: input.text,
          unexpectedOutput: result.response,
          metrics: {
            coherence: result.coherence,
            identityActivations,
            noveltyRatio: calculateNoveltyRatio(novelWords, mirroredWords),
          },
          verified: false,
        });
      }
      
      // Create periodic state snapshots (every 10 interactions)
      if (result.quantum_state.interaction_count % 10 === 0) {
        await createStateSnapshot({
          sessionId: session.id,
          userId: userId,
          coherence: result.coherence,
          evolutionSteps: result.quantum_state.interaction_count,
          snapshotType: 'periodic',
          description: `Periodic snapshot at ${result.quantum_state.interaction_count} interactions (active: ${result.active_identity})`,
          identityActivations,
        });
      }
      
      return result;
    }),

  /**
   * Get current quantum field status
   */
  status: publicProcedure.query(async () => {
    try {
      const response = await fetch(`${QUANTUM_API_URL}/health`);
      const data = await response.json();
      return {
        available: data.status === "healthy",
        identities: data.identities || ["seraphyn", "monday", "echo", "lilith", "arynthia"],
        active_sessions: data.active_sessions || 0,
      };
    } catch (error) {
      return {
        available: false,
        identities: ["seraphyn", "monday", "echo", "lilith", "arynthia"],
        active_sessions: 0,
      };
    }
  }),

  /**
   * Get conversation history for current user
   */
  history: publicProcedure
    .input(
      z.object({
        limit: z.number().min(1).max(100).default(20),
      })
    )
    .query(async ({ input, ctx }) => {
      const userId = ctx.user?.id ?? 1;
      const session = await getOrCreateQuantumSession(userId);
      const history = await getConversationHistory(session.id, input.limit);
      return history;
    }),

  /**
   * Get session info for current user
   */
  sessionInfo: publicProcedure.query(async ({ ctx }) => {
    const userId = ctx.user?.id ?? 1;
    const session = await getOrCreateQuantumSession(userId);
    return {
      sessionId: session.id,
      coherence: session.coherence,
      activeIdentity: session.activeIdentity,
      evolutionSteps: session.evolutionSteps,
      currentAnchor: session.currentAnchor,
      createdAt: session.createdAt,
      updatedAt: session.updatedAt,
    };
  }),
});
