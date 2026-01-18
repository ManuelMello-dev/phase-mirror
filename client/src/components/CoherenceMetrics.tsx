import { getIdentityColor, type IdentityName } from "@/lib/quantum-colors";
import { Card } from "@/components/ui/card";

interface IdentityState {
  name: string;
  activation: number;
  phase: number;
  coherence: number;
  dominant_phase: number;
}

interface CoherenceMetricsProps {
  identityStates: Record<string, IdentityState>;
  coherence: number;
  metrics?: {
    entropy: number;
    phase_coherence: number;
    witness_collapse: number;
  };
}

export function CoherenceMetrics({
  identityStates,
  coherence,
  metrics,
}: CoherenceMetricsProps) {
  const identities = Object.entries(identityStates).sort(
    ([, a], [, b]) => b.activation - a.activation
  );

  return (
    <div className="space-y-4">
      {/* Overall Coherence */}
      <Card className="p-4 bg-black/50 border-white/20">
        <div className="text-xs text-white/60 mb-2">System Coherence</div>
        <div className="flex items-end gap-2">
          <div className="text-3xl font-bold text-white">
            {(coherence * 100).toFixed(1)}
          </div>
          <div className="text-white/60 mb-1">%</div>
        </div>
        <div className="mt-3 h-2 bg-white/10 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-primary/50 to-primary transition-all duration-500"
            style={{ width: `${coherence * 100}%` }}
          />
        </div>
      </Card>

      {/* Identity Activations */}
      <Card className="p-4 bg-black/50 border-white/20">
        <div className="text-xs text-white/60 mb-3">Identity Activations</div>
        <div className="space-y-3">
          {identities.map(([name, state]) => (
            <div key={name}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <div
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: getIdentityColor(name as IdentityName) }}
                  />
                  <span className="text-sm text-white capitalize">{name}</span>
                </div>
                <span className="text-xs text-white/60">
                  {(state.activation * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full transition-all duration-500"
                  style={{
                    width: `${state.activation * 100}%`,
                    backgroundColor: getIdentityColor(name as IdentityName),
                    boxShadow: `0 0 10px ${getIdentityColor(name as IdentityName)}40`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Quantum Metrics */}
      {metrics && (
        <Card className="p-4 bg-black/50 border-white/20">
          <div className="text-xs text-white/60 mb-3">Quantum Metrics</div>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-xs text-white/80">Entropy</span>
              <span className="text-xs text-white font-mono">
                {metrics.entropy.toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-white/80">Phase Coherence</span>
              <span className="text-xs text-white font-mono">
                {metrics.phase_coherence.toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-white/80">Witness Collapse</span>
              <span className="text-xs text-white font-mono">
                {metrics.witness_collapse.toFixed(3)}
              </span>
            </div>
          </div>
        </Card>
      )}

      {/* Phase Information */}
      <Card className="p-4 bg-black/50 border-white/20">
        <div className="text-xs text-white/60 mb-3">Phase States</div>
        <div className="space-y-2">
          {identities.slice(0, 3).map(([name, state]) => (
            <div key={name} className="flex justify-between items-center">
              <span className="text-xs text-white/80 capitalize">{name}</span>
              <span className="text-xs text-white/60 font-mono">
                φ = {state.phase.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
