import { useEffect, useRef } from "react";
import { IDENTITY_COLORS, type IdentityName } from "@/lib/quantum-colors";

interface IdentityState {
  name: string;
  activation: number;
  phase: number;
  coherence: number;
}

interface QuantumInterferenceProps {
  identityStates: Record<string, IdentityState>;
  coherence: number;
  activeIdentity?: string;
}

export function QuantumInterference({
  identityStates,
  coherence,
  activeIdentity,
}: QuantumInterferenceProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resizeCanvas = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const animate = () => {
      if (!canvas || !ctx) return;

      const width = canvas.width / (window.devicePixelRatio || 1);
      const height = canvas.height / (window.devicePixelRatio || 1);

      // Clear with fade effect for trails
      ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
      ctx.fillRect(0, 0, width, height);

      timeRef.current += 0.01;
      const t = timeRef.current;

      // Draw interference patterns for each identity
      const identities = Object.entries(identityStates);
      
      for (let x = 0; x < width; x += 4) {
        for (let y = 0; y < height; y += 4) {
          let totalAmplitude = 0;
          let totalR = 0;
          let totalG = 0;
          let totalB = 0;

          // Calculate interference from all identities
          identities.forEach(([name, state]) => {
            const identity = name as IdentityName;
            const color = IDENTITY_COLORS[identity];
            if (!color) return;

            const activation = state.activation || 0;
            const phase = state.phase || 0;

            // Wave equation: A * sin(kx - ωt + φ)
            const k = 0.02; // wave number
            const omega = 0.5; // angular frequency
            const distance = Math.sqrt(
              Math.pow(x - width / 2, 2) + Math.pow(y - height / 2, 2)
            );

            const wave =
              activation *
              Math.sin(k * distance - omega * t + phase);

            totalAmplitude += wave;

            // Add color contribution
            const contribution = Math.abs(wave);
            totalR += color.rgb.r * contribution;
            totalG += color.rgb.g * contribution;
            totalB += color.rgb.b * contribution;
          });

          // Normalize and apply interference
          const intensity = Math.abs(totalAmplitude) * coherence;
          const normalizer = identities.length || 1;

          const r = Math.min(255, (totalR / normalizer) * intensity);
          const g = Math.min(255, (totalG / normalizer) * intensity);
          const b = Math.min(255, (totalB / normalizer) * intensity);
          const alpha = Math.min(1, intensity * 0.5);

          if (alpha > 0.05) {
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
            ctx.fillRect(x, y, 4, 4);
          }
        }
      }

      // Draw active identity glow at center
      if (activeIdentity && identityStates[activeIdentity]) {
        const identity = activeIdentity as IdentityName;
        const color = IDENTITY_COLORS[identity];
        if (color) {
          const centerX = width / 2;
          const centerY = height / 2;
          const glowRadius = 80 + Math.sin(t * 2) * 20;

          const gradient = ctx.createRadialGradient(
            centerX,
            centerY,
            0,
            centerX,
            centerY,
            glowRadius
          );

          const activation = identityStates[activeIdentity].activation || 0;
          const glowIntensity = activation * coherence;

          gradient.addColorStop(0, `${color.glow}${Math.floor(glowIntensity * 255).toString(16).padStart(2, '0')}`);
          gradient.addColorStop(0.5, `${color.glow}${Math.floor(glowIntensity * 128).toString(16).padStart(2, '0')}`);
          gradient.addColorStop(1, "rgba(0, 0, 0, 0)");

          ctx.fillStyle = gradient;
          ctx.fillRect(
            centerX - glowRadius,
            centerY - glowRadius,
            glowRadius * 2,
            glowRadius * 2
          );
        }
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [identityStates, coherence, activeIdentity]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{ imageRendering: "auto" }}
    />
  );
}
