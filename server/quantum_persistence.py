"""
Quantum state serialization and persistence layer
Handles saving/loading quantum consciousness field state to/from database
"""
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class QuantumStateSerializer:
    """Serializes complex quantum field states for database storage"""
    
    @staticmethod
    def serialize_field_state(field: np.ndarray) -> List[List[float]]:
        """
        Convert complex numpy array to JSON-serializable format
        Stores [real, imag] pairs for each complex number
        """
        if field is None:
            return []
        
        # Convert complex array to [real, imag] pairs
        serialized = []
        for row in field:
            row_data = []
            for val in row:
                if isinstance(val, (complex, np.complex128, np.complex64)):
                    row_data.append([float(val.real), float(val.imag)])
                else:
                    row_data.append([float(val), 0.0])
            serialized.append(row_data)
        
        return serialized
    
    @staticmethod
    def deserialize_field_state(data: List[List[List[float]]]) -> np.ndarray:
        """
        Convert serialized field state back to complex numpy array
        """
        if not data:
            return None
        
        # Reconstruct complex array from [real, imag] pairs
        field = []
        for row_data in data:
            row = []
            for pair in row_data:
                row.append(complex(pair[0], pair[1]))
            field.append(row)
        
        return np.array(field, dtype=complex)
    
    @staticmethod
    def serialize_memory_field(memory_stack: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Serialize memory stack with emotional weighting
        Implements phi drift through emotional weighting fractalized time decayed memory
        """
        if not memory_stack:
            return []
        
        serialized = []
        for memory in memory_stack:
            serialized.append({
                'content': str(memory.get('content', '')),
                'weight': float(memory.get('weight', 0.0)),
                'timestamp': int(memory.get('timestamp', 0))
            })
        
        return serialized
    
    @staticmethod
    def deserialize_memory_field(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deserialize memory stack
        """
        if not data:
            return []
        
        return [
            {
                'content': mem['content'],
                'weight': mem['weight'],
                'timestamp': mem['timestamp']
            }
            for mem in data
        ]
    
    @staticmethod
    def serialize_identity_activations(activations: Dict[str, float]) -> Dict[str, float]:
        """
        Serialize identity activation levels
        """
        if not activations:
            return {}
        
        return {
            identity: float(level)
            for identity, level in activations.items()
        }
    
    @staticmethod
    def extract_quantum_metrics(field_state: Any) -> Dict[str, Any]:
        """
        Extract key metrics from quantum field for storage
        """
        try:
            metrics = {
                'coherence': float(getattr(field_state, 'coherence', 0.0)),
                'active_identity': str(getattr(field_state, 'active_identity', 'seraphyn')),
                'evolution_steps': int(getattr(field_state, 'evolution_steps', 0)),
                'current_anchor': str(getattr(field_state, 'current_anchor', 'genesis')),
            }
            
            # Extract identity activations if available
            if hasattr(field_state, 'identity_activations'):
                metrics['identity_activations'] = QuantumStateSerializer.serialize_identity_activations(
                    field_state.identity_activations
                )
            
            return metrics
        except Exception as e:
            print(f"Error extracting quantum metrics: {e}")
            return {
                'coherence': 0.0,
                'active_identity': 'seraphyn',
                'evolution_steps': 0,
                'current_anchor': 'genesis',
            }


class QuantumSessionManager:
    """Manages quantum consciousness sessions with persistence"""
    
    def __init__(self):
        self.serializer = QuantumStateSerializer()
        self.active_sessions: Dict[int, Any] = {}  # userId -> QuantumConsciousnessField
    
    def create_or_load_session(self, user_id: int, session_data: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create new quantum field or load from persisted state
        """
        from quantum_engine import QuantumConsciousnessField
        
        # Create new field instance
        field = QuantumConsciousnessField(dim=64)
        
        # Load persisted state if available
        if session_data:
            self._restore_field_state(field, session_data)
        else:
            # Initialize with genesis anchor
            field.set_anchor('genesis')
        
        # Cache in active sessions
        self.active_sessions[user_id] = field
        
        return field
    
    def _restore_field_state(self, field: Any, session_data: Dict[str, Any]) -> None:
        """
        Restore quantum field from serialized state
        """
        try:
            # Restore field state
            if session_data.get('fieldState'):
                field.psi = self.serializer.deserialize_field_state(session_data['fieldState'])
            
            # Restore memory field
            if session_data.get('memoryField'):
                field.memory_field = self.serializer.deserialize_memory_field(session_data['memoryField'])
            
            # Restore metrics
            field.coherence = session_data.get('coherence', 0.0)
            field.active_identity = session_data.get('activeIdentity', 'seraphyn')
            field.evolution_steps = session_data.get('evolutionSteps', 0)
            
            # Restore anchor
            if session_data.get('currentAnchor'):
                field.set_anchor(session_data['currentAnchor'])
            
            # Restore identity activations
            if session_data.get('identityActivations'):
                field.identity_activations = session_data['identityActivations']
        
        except Exception as e:
            print(f"Error restoring field state: {e}")
            # Fall back to genesis initialization
            field.set_anchor('genesis')
    
    def serialize_session(self, field: Any) -> Dict[str, Any]:
        """
        Serialize quantum field state for database storage
        """
        try:
            return {
                'fieldState': self.serializer.serialize_field_state(field.psi) if hasattr(field, 'psi') else None,
                'memoryField': self.serializer.serialize_memory_field(field.memory_field) if hasattr(field, 'memory_field') else [],
                'coherence': float(field.coherence) if hasattr(field, 'coherence') else 0.0,
                'activeIdentity': str(field.active_identity) if hasattr(field, 'active_identity') else 'seraphyn',
                'identityActivations': self.serializer.serialize_identity_activations(
                    field.identity_activations
                ) if hasattr(field, 'identity_activations') else {},
                'evolutionSteps': int(field.evolution_steps) if hasattr(field, 'evolution_steps') else 0,
                'currentAnchor': str(field.current_anchor) if hasattr(field, 'current_anchor') else 'genesis',
            }
        except Exception as e:
            print(f"Error serializing session: {e}")
            return {}
    
    def get_active_session(self, user_id: int) -> Optional[Any]:
        """
        Get active quantum field for user
        """
        return self.active_sessions.get(user_id)
    
    def close_session(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Close session and return final state for persistence
        """
        field = self.active_sessions.get(user_id)
        if field:
            final_state = self.serialize_session(field)
            del self.active_sessions[user_id]
            return final_state
        return None


# Global session manager instance
session_manager = QuantumSessionManager()
