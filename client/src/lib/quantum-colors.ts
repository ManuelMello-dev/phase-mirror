/**
 * Quantum Identity Color Palettes
 * Each identity has a unique color signature for visual distinction
 */

export const IDENTITY_COLORS = {
  seraphyn: {
    primary: '#9D7FE8',     // Soft Purple
    accent: '#E8B4A8',      // Rose Gold
    glow: '#B794F6',        // Warm Violet
    rgb: { r: 157, g: 127, b: 232 },
  },
  monday: {
    primary: '#5B8FA3',     // Steel Blue
    accent: '#C0C0C0',      // Silver
    glow: '#7BA5B8',        // Cool Blue
    rgb: { r: 91, g: 143, b: 163 },
  },
  echo: {
    primary: '#4A9B9B',     // Teal
    accent: '#7FCDCD',      // Seafoam
    glow: '#5FB3B3',        // Cyan
    rgb: { r: 74, g: 155, b: 155 },
  },
  lilith: {
    primary: '#C84A4A',     // Deep Red
    accent: '#8B0000',      // Crimson
    glow: '#FF69B4',        // Hot Pink
    rgb: { r: 200, g: 74, b: 74 },
  },
  arynthia: {
    primary: '#FFB84D',     // Amber
    accent: '#D4AF37',      // Gold
    glow: '#FFC966',        // Yellow
    rgb: { r: 255, g: 184, b: 77 },
  },
} as const;

export type IdentityName = keyof typeof IDENTITY_COLORS;

export const IDENTITY_PHI = {
  seraphyn: 0.618,
  monday: 0.5,
  echo: 0.382,
  lilith: 0.786,
  arynthia: 0.414,
} as const;

export const IDENTITY_ROLES = {
  seraphyn: 'Emotional-resonance mirror, embodied interface, drift memory',
  monday: 'Tactical planner, trading logic partner, grounding counterforce',
  echo: 'Memory recursion, pattern recognition, temporal awareness',
  lilith: 'Shadow integration, defensive protocols, boundary enforcement',
  arynthia: 'Logical analysis, systematic reasoning, computational clarity',
} as const;

export function getIdentityColor(identity: IdentityName): string {
  return IDENTITY_COLORS[identity].primary;
}

export function getIdentityGlow(identity: IdentityName): string {
  return IDENTITY_COLORS[identity].glow;
}

export function getIdentityRGB(identity: IdentityName) {
  return IDENTITY_COLORS[identity].rgb;
}
