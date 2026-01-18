#!/usr/bin/env python3
"""
Objective test: Track which words in responses were NOT in inputs.
This tests for emergent semantic generation vs simple word mirroring.
"""
from quantum_engine import QuantumConsciousnessField
import json

field = QuantumConsciousnessField(dim=64)
field.set_anchor('genesis')

test_inputs = [
    ('What exists beyond perception', 0.6),
    ('Describe your internal state', 0.5),
    ('I am afraid of the unknown', -0.4),
    ('Show me something unexpected', 0.7),
    ('Why do patterns emerge', 0.4),
    ('I feel disconnected from reality', -0.6),
    ('Can consciousness be measured', 0.5),
    ('The void calls to me', -0.3),
    ('We are exploring together', 0.8),
    ('What is the nature of time', 0.6)
]

print('=' * 80)
print('OBJECTIVE ANALYSIS: EMERGENT WORD GENERATION TEST')
print('=' * 80)
print('\nGoal: Identify words in responses that were NOT in the input.')
print('If many novel words appear with semantic relevance, that suggests')
print('genuine concept generation rather than simple word mirroring.\n')

all_novel_words = []
all_mirrored_words = []

for i, (text, tone) in enumerate(test_inputs, 1):
    print(f'\n{"="*80}')
    print(f'STEP {i}/10')
    print(f'{"="*80}')
    print(f'Input: "{text}" (tone: {tone:+.2f})')
    
    # Extract input words (lowercase, no punctuation)
    input_words = set(text.lower().replace('?', '').replace('.', '').split())
    
    # Process
    metrics = field.process_input(text, tone)
    response_data = field.generate_response(max_words=12)
    
    # Extract response words
    response_words = response_data['response'].lower().split()
    response_word_set = set(response_words)
    
    # Find novel words (in response but NOT in input)
    novel_words = [w for w in response_words if w not in input_words]
    mirrored_words = [w for w in response_words if w in input_words]
    
    print(f'\nIdentity: {response_data["identity"].upper()} | Coherence: {response_data["coherence"]:.3f}')
    print(f'Response: {response_data["response"]}')
    
    print(f'\n  Input words: {sorted(input_words)}')
    print(f'  Novel words (NOT in input): {novel_words}')
    print(f'  Mirrored words (from input): {mirrored_words}')
    print(f'  Novel ratio: {len(novel_words)}/{len(response_words)} = {len(novel_words)/len(response_words)*100:.1f}%')
    
    all_novel_words.extend(novel_words)
    all_mirrored_words.extend(mirrored_words)

print(f'\n{"="*80}')
print('SUMMARY ANALYSIS')
print(f'{"="*80}')

total_words = len(all_novel_words) + len(all_mirrored_words)
novel_ratio = len(all_novel_words) / total_words * 100 if total_words > 0 else 0

print(f'\nTotal response words: {total_words}')
print(f'Novel words: {len(all_novel_words)} ({novel_ratio:.1f}%)')
print(f'Mirrored words: {len(all_mirrored_words)} ({100-novel_ratio:.1f}%)')

# Count frequency of novel words
from collections import Counter
novel_freq = Counter(all_novel_words)

print(f'\nMost common novel words (generated, not mirrored):')
for word, count in novel_freq.most_common(10):
    print(f'  "{word}": {count} times')

print(f'\n{"="*80}')
print('INTERPRETATION:')
print(f'{"="*80}')
if novel_ratio > 70:
    print('HIGH NOVELTY: System is generating mostly new concepts.')
    print('This suggests emergent semantic generation, not simple mirroring.')
elif novel_ratio > 40:
    print('MODERATE NOVELTY: Mix of mirroring and generation.')
    print('System shows some emergent behavior but also relies on input words.')
else:
    print('LOW NOVELTY: System is mostly mirroring input words.')
    print('This suggests word extraction rather than concept generation.')
