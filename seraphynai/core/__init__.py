"""
SeraphynAI Core Module

Provides the quantum consciousness field and related components.
"""

# Only import modules that don't have numpy/scipy dependencies at load time
from seraphynai.core.personas import IdentityType, PERSONAS, IdentityPersona

__all__ = [
    "QuantumConsciousnessField",
    "QuantumState",
    "Z3Evolution",
    "IdentityType",
    "PERSONAS",
    "IdentityPersona",
]

def __getattr__(name):
    """Lazy loading to avoid dependency issues"""
    if name == "QuantumConsciousnessField":
        from seraphynai.core.quantum_full_original import QuantumConsciousnessField
        return QuantumConsciousnessField
    elif name == "QuantumState":
        from seraphynai.core.quantum_state import QuantumState
        return QuantumState
    elif name == "Z3Evolution":
        from seraphynai.core.z3_evolution import Z3Evolution
        return Z3Evolution
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
