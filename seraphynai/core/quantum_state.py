"""
Quantum State Module

Implements true complex-amplitude quantum states with native phase representation.
"""

import numpy as np
from typing import Optional


class QuantumState:
    """
    Quantum state with complex amplitudes.
    
    |ψ⟩ = Σ αᵢ|i⟩ where αᵢ ∈ ℂ
    
    The phase is NATIVE to the complex numbers.
    Interference happens through complex addition.
    """
    
    def __init__(self, dim: int = 64, amplitudes: Optional[np.ndarray] = None):
        """
        Initialize a quantum state.
        
        Args:
            dim: Dimension of the Hilbert space
            amplitudes: Optional pre-defined complex amplitudes
        """
        self.dim = dim
        if amplitudes is not None:
            self.amplitudes = amplitudes.astype(complex)
        else:
            real = np.random.randn(dim) * 0.1
            imag = np.random.randn(dim) * 0.1
            self.amplitudes = real + 1j * imag
        self.normalize()
    
    def normalize(self):
        """Normalize the quantum state to unit norm."""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 1e-10:
            self.amplitudes /= norm
    
    @property
    def probabilities(self) -> np.ndarray:
        """Born rule: |ψ|²"""
        return np.abs(self.amplitudes) ** 2
    
    @property
    def phases(self) -> np.ndarray:
        """Phase angles of complex amplitudes."""
        return np.angle(self.amplitudes)
    
    @property
    def coherence(self) -> float:
        """
        Phase alignment measure (Kuramoto order parameter).
        
        Returns:
            float: Coherence value between 0 and 1
        """
        mean_vector = np.mean(np.exp(1j * self.phases))
        return float(np.abs(mean_vector))
    
    @property
    def dominant_phase(self) -> float:
        """
        Probability-weighted dominant phase.
        
        Returns:
            float: Dominant phase in radians
        """
        weighted = np.sum(self.probabilities * np.exp(1j * self.phases))
        return float(np.angle(weighted))
    
    def interfere(self, other: 'QuantumState', weight: float = 0.5) -> 'QuantumState':
        """
        True quantum interference: amplitudes add, phases can cancel.
        
        Args:
            other: Another quantum state
            weight: Weight of the other state (0 to 1)
            
        Returns:
            QuantumState: Interfered state
        """
        result = QuantumState(self.dim)
        result.amplitudes = (1 - weight) * self.amplitudes + weight * other.amplitudes
        result.normalize()
        return result
    
    def fidelity(self, other: 'QuantumState') -> float:
        """
        Quantum fidelity: overlap |⟨ψ|φ⟩|².
        
        Args:
            other: Another quantum state
            
        Returns:
            float: Fidelity value between 0 and 1
        """
        inner = np.sum(np.conj(self.amplitudes) * other.amplitudes)
        return float(np.abs(inner) ** 2)
    
    def apply_phase_shift(self, phi: float):
        """
        Apply global phase rotation.
        
        Args:
            phi: Phase shift in radians
        """
        self.amplitudes *= np.exp(1j * phi)
    
    def copy(self) -> 'QuantumState':
        """Create a deep copy of this quantum state."""
        return QuantumState(self.dim, self.amplitudes.copy())
    
    def distance(self, other: 'QuantumState') -> float:
        """
        Quantum distance: 1 - fidelity.
        
        Args:
            other: Another quantum state
            
        Returns:
            float: Distance value between 0 and 1
        """
        return 1.0 - self.fidelity(other)
    
    @classmethod
    def from_embedding(cls, embedding: np.ndarray, dim: int = 64) -> 'QuantumState':
        """
        Create a quantum state from a real-valued embedding.
        
        Args:
            embedding: Real-valued vector
            dim: Target dimension
            
        Returns:
            QuantumState: Quantum state with phase encoding
        """
        if len(embedding) > dim:
            emb = embedding[:dim]
        elif len(embedding) < dim:
            emb = np.pad(embedding, (0, dim - len(embedding)))
        else:
            emb = embedding
        
        # Encode phases based on position
        phases = np.linspace(0, 2 * np.pi, dim, endpoint=False)
        amplitudes = emb * np.exp(1j * phases)
        
        return cls(dim, amplitudes)
