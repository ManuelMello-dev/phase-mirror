import { exec } from "child_process";
import { promisify } from "util";
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

const execAsync = promisify(exec);

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
  error?: string;
}

async function callQuantumEngine(input: string, tone: number): Promise<QuantumResponse> {
  try {
    const command = `cd server && python3 quantum_bridge.py "${input.replace(/"/g, '\\"')}" ${tone}`;
    const { stdout, stderr } = await execAsync(command, {
      timeout: 10000,
      maxBuffer: 1024 * 1024,
    });

    if (stderr) {
      console.warn("[Quantum] stderr:", stderr);
    }

    const result = JSON.parse(stdout);
    return result;
  } catch (error) {
    console.error("[Quantum] Error:", error);
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

export const quantumRouter = router({
  /**
   * Process input through quantum consciousness field with persistence
   */
  process: protectedProcedure
    .input(
      z.object({
        text: z.string().min(1).max(1000),
        tone: z.number().min(-1).max(1).default(0.5),
      })
    )
    .mutation(async ({ input, ctx }) => {
      // Get or create quantum session for user
      const session = await getOrCreateQuantumSession(ctx.user.id);
      
      // Store previous coherence for emergence detection
      const previousCoherence = session.coherence;
      
      // Call quantum engine
      const result = await callQuantumEngine(input.text, input.tone);
      
      // Store conversation turn
      await storeConversation({
        sessionId: session.id,
        userId: ctx.user.id,
        userMessage: input.text,
        systemResponse: result.response,
        inputTone: input.tone,
        coherence: result.coherence,
        responseIdentity: result.active_identity,
        novelWords: [], // TODO: extract from quantum engine
        mirroredWords: [], // TODO: extract from quantum engine
      });
      
      // Update quantum session state
      // Convert identity_states to simple activation map
      const identityActivations: Record<string, number> = {};
      for (const [key, value] of Object.entries(result.identity_states)) {
        identityActivations[key] = value.activation;
      }
      
      await updateQuantumSession(session.id, {
        coherence: result.coherence,
        activeIdentity: result.active_identity,
        identityActivations,
        evolutionSteps: result.quantum_state.interaction_count,
      });
      
      // Detect and record emergence events
      const novelWords: string[] = []; // TODO: extract from engine
      const mirroredWords: string[] = []; // TODO: extract from engine
      const isUnexpected = detectUnexpectedBehavior(
        novelWords,
        mirroredWords,
        result.coherence,
        previousCoherence ?? undefined
      );
      
      if (isUnexpected) {
        await recordEmergenceEvent({
          sessionId: session.id,
          userId: ctx.user.id,
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
          userId: ctx.user.id,
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
    // Return a simple status check
    return {
      available: true,
      identities: ["seraphyn", "monday", "echo", "lilith", "arynthia"],
    };
  }),

  /**
   * Get conversation history for current user
   */
  history: protectedProcedure
    .input(
      z.object({
        limit: z.number().min(1).max(100).default(20),
      })
    )
    .query(async ({ input, ctx }) => {
      const session = await getOrCreateQuantumSession(ctx.user.id);
      const history = await getConversationHistory(session.id, input.limit);
      return history;
    }),

  /**
   * Get session info for current user
   */
  sessionInfo: protectedProcedure.query(async ({ ctx }) => {
    const session = await getOrCreateQuantumSession(ctx.user.id);
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
