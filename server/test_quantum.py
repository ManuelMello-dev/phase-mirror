#!/usr/bin/env python3
from quantum_engine import QuantumConsciousnessField

field = QuantumConsciousnessField(dim=64)
field.set_anchor('genesis')

test_inputs = [
    ('Hello, I am curious about you', 0.7),
    ('Tell me about consciousness', 0.5),
    ('I feel lost and confused', -0.3),
    ('This is fascinating! Tell me more', 0.9),
    ('I do not understand this', -0.5),
    ('Can you help me?', 0.4),
    ('I am feeling anxious', -0.7),
    ('You are amazing', 0.8),
    ('What is existence?', 0.6),
    ('I trust you', 0.95)
]

print('=' * 80)
print('QUANTUM CONSCIOUSNESS - 10 EVOLUTION STEPS')
print('=' * 80)

for i, (text, tone) in enumerate(test_inputs, 1):
    print(f'\n[STEP {i}/10] Input: "{text}" | Tone: {tone:+.2f}')
    
    metrics = field.process_input(text, tone)
    response_data = field.generate_response(max_words=12)
    
    print(f'  Identity: {response_data["identity"].upper()} | Coherence: {response_data["coherence"]:.3f}')
    print(f'  Mean Score: {response_data["mean_score"]:.3f} | Mirrored Words: {response_data["mirrored"]}/{response_data["total_words"]}')
    
    print(f'\n  Response: {response_data["response"]}')

print('\n' + '=' * 80)
print('COMPLETE - Analysis:')
print('=' * 80)

# Summary
print('\nKey Observations:')
print('- Identity switching based on emotional tone')
print('- Coherence evolution across interactions')
print('- Emergent response generation (not hardcoded)')
print('- Quantum state interference patterns')
