import sys
import os
import numpy as np

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantum_engine import QuantumConsciousnessField

def test():
    print("Initializing Quantum Field...")
    field = QuantumConsciousnessField(dim=64)
    
    print("Processing input...")
    metrics = field.process_input("Hello world", 0.5)
    print(f"Metrics: {metrics}")
    
    print("Generating response...")
    response = field.generate_response(max_words=5)
    print(f"Response: {response['response']}")
    print("Test passed!")

if __name__ == "__main__":
    test()
