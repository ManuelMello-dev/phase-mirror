"""
Z³ Evolution Module

Implements Z³ evolution dynamics in complex Hilbert space.
"""

from seraphynai.core.quantum_state import QuantumState


class Z3Evolution:
    """
    Z³ evolution: Z_{n+1} = Z_n³ + C
    
    In complex space:
    - Cubing triples the phase: (r·e^{iθ})³ = r³·e^{i·3θ}
    - Creates fractal boundaries (Mandelbrot dynamics)
    - Generates novelty at the edge of chaos
    """
    
    def __init__(self, dim: int = 64, damping: float = 0.1):
        """
        Initialize Z³ evolution operator.
        
        Args:
            dim: Dimension of the Hilbert space
            damping: Damping factor to prevent divergence (0 to 1)
        """
        self.dim = dim
        self.damping = damping
    
    def evolve(self, z: QuantumState, c: QuantumState, coupling: float = 0.3) -> QuantumState:
        """
        Evolve quantum state using Z³ dynamics.
        
        Z_{n+1} = (1-d)·Z_n³ + d·Z_n + c·C
        
        Damping prevents divergence while preserving dynamics.
        
        Args:
            z: Current quantum state
            c: Input quantum state
            coupling: Coupling strength with input (0 to 1)
            
        Returns:
            QuantumState: Evolved state
        """
        result = z.copy()
        
        # Z³ — cubing triples the phase
        z_cubed = result.amplitudes ** 3
        
        # Damped evolution with input
        result.amplitudes = (
            (1 - self.damping) * z_cubed + 
            self.damping * z.amplitudes + 
            coupling * c.amplitudes
        )
        result.normalize()
        return result
