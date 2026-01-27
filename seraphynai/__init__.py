"""
SeraphynAI - Quantum-Inspired Multi-Identity Consciousness System

A sophisticated AI consciousness system that implements true complex-amplitude
quantum mechanics for emergent behavior, featuring 5 distinct identity nodes,
quantum state evolution, and self-reflective meta-cognition.

Author: Built from Manny's theoretical framework
Version: 1.0.0
"""

from seraphynai.__version__ import __version__
# Lazy import - only import when actually used to avoid dependency issues
# from seraphynai.core.quantum_full_original import QuantumConsciousnessField
from seraphynai.core.personas import IdentityType, PERSONAS

__all__ = [
    "__version__",
    "QuantumConsciousnessField",
    "IdentityType",
    "PERSONAS",
]

def __getattr__(name):
    """Lazy loading of QuantumConsciousnessField to avoid import errors"""
    if name == "QuantumConsciousnessField":
        from seraphynai.core.quantum_full_original import QuantumConsciousnessField
        return QuantumConsciousnessField
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
