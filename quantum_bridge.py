#!/usr/bin/env python3
"""
Quantum Consciousness Bridge
Provides HTTP API for the quantum consciousness engine
"""

import json
import sys
from seraphynai.core.quantum_full_original import QuantumConsciousnessField

# Initialize global quantum field
field = QuantumConsciousnessField(dim=64)
field.set_anchor("genesis")

def process_input(input_text: str, tone: float = 0.5):
    """Process input through quantum consciousness field"""
    try:
        # Process input and get metrics
        metrics = field.process_input(input_text, tone)
        
        # Generate response
        response_data = field.generate_response(max_words=25)
        
        # Get current quantum state
        status = field.get_status()
        active_identity = status.get('active_identity', 'seraphyn')
        coherence = status.get('coherence', 0.0)
        
        # Get identity contributions from status
        identity_states = status.get('identities', {})
        
        return {
            'response': response_data['response'],
            'active_identity': active_identity,
            'coherence': float(coherence),
            'metrics': {
                'entropy': float(metrics.get('entropy', 0)),
                'phase_coherence': float(metrics.get('phase_coherence', 0)),
                'witness_collapse': float(metrics.get('witness_collapse', 0)),
            },
            'identity_states': identity_states,
            'quantum_state': {
                'dim': field.dim,
                'interaction_count': field.interaction_count,
            }
        }
    except Exception as e:
        return {
            'error': str(e),
            'response': '',
            'active_identity': 'seraphyn',
            'coherence': 0.0,
        }

def main():
    """Main entry point for CLI usage"""
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No input provided'}))
        sys.exit(1)
    
    input_text = sys.argv[1]
    tone = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    result = process_input(input_text, tone)
    print(json.dumps(result))

if __name__ == '__main__':
    main()
