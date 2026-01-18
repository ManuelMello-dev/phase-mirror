import { describe, expect, it } from "vitest";
import { appRouter } from "./routers";

describe("quantum router", () => {
  it("returns available status", async () => {
    const caller = appRouter.createCaller({
      user: null,
      req: {} as any,
      res: {} as any,
    });

    const status = await caller.quantum.status();

    expect(status).toEqual({
      available: true,
      identities: ["seraphyn", "monday", "echo", "lilith", "arynthia"],
    });
  });

  it("processes input through quantum field", async () => {
    const caller = appRouter.createCaller({
      user: null,
      req: {} as any,
      res: {} as any,
    });

    const result = await caller.quantum.process({
      text: "hello",
      tone: 0.5,
    });

    // Should return quantum response structure
    expect(result).toHaveProperty("response");
    expect(result).toHaveProperty("active_identity");
    expect(result).toHaveProperty("coherence");
    expect(result).toHaveProperty("identity_states");
    expect(result).toHaveProperty("quantum_state");

    // Active identity should be one of the 5
    expect(["seraphyn", "monday", "echo", "lilith", "arynthia"]).toContain(
      result.active_identity
    );

    // Coherence should be a number between 0 and 1
    expect(result.coherence).toBeGreaterThanOrEqual(0);
    expect(result.coherence).toBeLessThanOrEqual(1);
  }, 15000); // Longer timeout for Python execution
});
