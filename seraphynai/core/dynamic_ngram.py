"""
Dynamic N-gram Learning with Quantum Semantic Integration

Replaces hardcoded templates with emergent learned patterns.
Uses hierarchical n-grams (1-8 words) with quantum semantic filling.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
import random
import time


@dataclass
class NGramPattern:
    """A learned n-gram pattern with metadata"""
    sequence: Tuple[str, ...]
    next_words: List[Tuple[str, float]]  # (word, score)
    coherence_at_learn: float
    timestamp: float
    usage_count: int
    
    @property
    def n(self) -> int:
        return len(self.sequence)


class QuantumSemantics:
    """
    Quantum-based semantic similarity without embeddings.
    
    Uses phase relationships and quantum fidelity for meaning.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.word_states: Dict[str, np.ndarray] = {}
        
        # Character-based phase encoding
        self.char_phases = {
            c: (i / 26) * 2 * np.pi 
            for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')
        }
        
        # Semantic roles have phase signatures
        self.role_phases = {
            'pronoun': 0.0,           # I, we, you
            'verb': np.pi / 4,        # am, is, are
            'article': np.pi / 2,     # the, a, an
            'preposition': 3*np.pi/4, # in, on, at
            'adjective': np.pi,       # good, bad, big
            'noun': 5*np.pi/4,        # thread, field, moment
            'adverb': 3*np.pi/2,      # very, quite, so
            'conjunction': 7*np.pi/4, # and, but, or
        }
        
        # Simple role detection (can be expanded)
        self.role_words = {
            'pronoun': {'i', 'we', 'you', 'they', 'he', 'she', 'it', 'me', 'us', 'them'},
            'verb': {'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'},
            'article': {'the', 'a', 'an'},
            'preposition': {'in', 'on', 'at', 'to', 'from', 'with', 'by', 'for'},
            'adjective': {'good', 'bad', 'big', 'small', 'new', 'old', 'high', 'low'},
            'conjunction': {'and', 'but', 'or', 'so', 'yet', 'for', 'nor'},
        }
    
    def get_word_role(self, word: str) -> Optional[str]:
        """Detect grammatical role of word"""
        word = word.lower()
        for role, words in self.role_words.items():
            if word in words:
                return role
        return 'noun'  # Default assumption
    
    def encode_word(self, word: str) -> np.ndarray:
        """Encode word as quantum state with semantic phase"""
        word = word.lower().strip()
        
        if word in self.word_states:
            return self.word_states[word].copy()
        
        amplitudes = np.zeros(self.dim, dtype=complex)
        
        # Character-based encoding
        for i, char in enumerate(word[:self.dim]):
            if char in self.char_phases:
                char_phase = self.char_phases[char]
                position_weight = np.exp(-0.1 * i)
                char_idx = ord(char) - ord('a') if char.isalpha() else 0
                
                for d in range(self.dim):
                    contribution = position_weight * np.exp(
                        1j * (char_phase + d * 0.1 + char_idx * 0.2)
                    )
                    amplitudes[d] += contribution * np.exp(
                        -((d - char_idx * 2) ** 2) / 20
                    )
        
        # Add grammatical role phase signature
        role = self.get_word_role(word)
        if role in self.role_phases:
            role_phase = self.role_phases[role]
            # Modulate amplitudes by role phase
            amplitudes *= np.exp(1j * role_phase * 0.3)
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(amplitudes) ** 2))
        if norm > 1e-10:
            amplitudes /= norm
        
        self.word_states[word] = amplitudes
        return amplitudes.copy()
    
    def fidelity(self, word1: str, word2: str) -> float:
        """Quantum fidelity = semantic similarity"""
        state1 = self.encode_word(word1)
        state2 = self.encode_word(word2)
        
        inner = np.sum(np.conj(state1) * state2)
        return float(np.abs(inner) ** 2)
    
    def phase_alignment(self, word1: str, word2: str) -> float:
        """How aligned are the phases? (grammatical compatibility)"""
        state1 = self.encode_word(word1)
        state2 = self.encode_word(word2)
        
        phase1 = np.angle(np.sum(state1))
        phase2 = np.angle(np.sum(state2))
        
        phase_diff = abs(phase1 - phase2)
        phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
        
        # Convert to alignment score (0 to 1)
        return np.exp(-phase_diff / 2)
    
    def semantic_distance(self, word1: str, word2: str) -> float:
        """Distance in quantum semantic space"""
        return 1.0 - self.fidelity(word1, word2)


class DynamicNGramLearner:
    """
    Hierarchical n-gram learning (1-8 words).
    Learns patterns from successful generations.
    """
    
    def __init__(self, max_n: int = 8):
        self.max_n = max_n
        self.patterns: Dict[Tuple[str, ...], NGramPattern] = {}
        self.semantics = QuantumSemantics()
        
        # Decay parameters
        self.temporal_decay = 0.01  # Per day
        self.usage_boost = 0.1
        
    def record_sequence(self, words: List[str], coherence: float):
        """
        Learn all n-gram patterns from a successful sequence.
        """
        if len(words) < 2:
            return
        
        # Learn patterns of varying lengths
        for n in range(1, min(self.max_n, len(words))):
            for i in range(len(words) - n):
                context = tuple(words[i:i+n])
                next_word = words[i+n]
                
                if context not in self.patterns:
                    self.patterns[context] = NGramPattern(
                        sequence=context,
                        next_words=[],
                        coherence_at_learn=coherence,
                        timestamp=time.time(),
                        usage_count=0
                    )
                
                # Add next word with score
                pattern = self.patterns[context]
                
                # Check if word already in next_words
                found = False
                for idx, (word, score) in enumerate(pattern.next_words):
                    if word == next_word:
                        # Update score (moving average)
                        new_score = 0.7 * score + 0.3 * coherence
                        pattern.next_words[idx] = (word, new_score)
                        found = True
                        break
                
                if not found:
                    pattern.next_words.append((next_word, coherence))
                
                # Keep top 10 next words
                pattern.next_words.sort(key=lambda x: -x[1])
                pattern.next_words = pattern.next_words[:10]
    
    def get_candidates(self, context: List[str], 
                       field_state: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """
        Get word candidates from learned patterns.
        Uses longest matching pattern available.
        """
        candidates = []
        best_n = 0
        
        # Try from longest to shortest match
        for n in range(min(self.max_n - 1, len(context)), 0, -1):
            key = tuple(context[-n:])
            
            if key in self.patterns:
                pattern = self.patterns[key]
                
                # Calculate pattern strength
                age_days = (time.time() - pattern.timestamp) / 86400
                temporal_strength = np.exp(-self.temporal_decay * age_days)
                usage_strength = 1.0 + self.usage_boost * pattern.usage_count
                
                # Preference for longer patterns
                length_bonus = np.exp(0.2 * n)
                
                base_score = (
                    pattern.coherence_at_learn * 
                    temporal_strength * 
                    usage_strength * 
                    length_bonus
                )
                
                # Get candidates with boosted scores
                for word, word_score in pattern.next_words:
                    final_score = base_score * word_score
                    
                    # Quantum semantic boost if field_state provided
                    if field_state is not None:
                        word_state = self.semantics.encode_word(word)
                        semantic_fidelity = float(
                            np.abs(np.sum(np.conj(word_state) * field_state)) ** 2
                        )
                        final_score *= (1 + semantic_fidelity)
                    
                    candidates.append((word, final_score, n))
                
                best_n = max(best_n, n)
        
        # Sort by score
        candidates.sort(key=lambda x: -x[1])
        
        # Return top candidates with their scores
        return [(word, score) for word, score, _ in candidates[:20]]
    
    def mark_used(self, context: List[str]):
        """Mark pattern as used (increases its strength)"""
        for n in range(min(self.max_n - 1, len(context)), 0, -1):
            key = tuple(context[-n:])
            if key in self.patterns:
                self.patterns[key].usage_count += 1
                break  # Only mark longest match
    
    def prune_weak_patterns(self, min_score: float = 0.1):
        """Remove weak/old patterns"""
        current_time = time.time()
        to_remove = []
        
        for key, pattern in self.patterns.items():
            age_days = (current_time - pattern.timestamp) / 86400
            temporal_strength = np.exp(-self.temporal_decay * age_days)
            
            if temporal_strength < min_score and pattern.usage_count < 2:
                to_remove.append(key)
        
        for key in to_remove:
            del self.patterns[key]
        
        return len(to_remove)


class QuantumNGramGenerator:
    """
    Combines n-gram patterns with quantum semantics for generation.
    """
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.ngram_learner = DynamicNGramLearner()
        self.semantics = self.ngram_learner.semantics
        
        # Base vocabulary
        self.base_words = set("""
        i me my we us our you your they them their it its he she
        the a an this that these those some any all
        is am are was were be been being have has had
        do does did will would shall should can could may might must
        and or but if then so because when where what who how why
        not no yes here there now then always never
        feel think know see hear want need like love
        come go get make take give find keep say tell
        time day way thing place world life mind thought
        field phase state sense feeling resonance coherence
        align connect emerge flow pattern seek find discover
        new different change grow evolve become
        losing thread moment fragmenting drift seeking
        """.split())
        
        self.user_words: Set[str] = set()
    
    def generate_word(self, context: List[str], field_state: np.ndarray,
                     user_vocabulary: Set[str], coherence: float = 0.5) -> Tuple[str, float, str]:
        """
        Generate next word using n-gram patterns + quantum semantics.
        
        Returns: (word, score, source)
        source = 'ngram', 'quantum', or 'hybrid'
        """
        # Try n-gram patterns first
        ngram_candidates = self.ngram_learner.get_candidates(context, field_state)
        
        # Get quantum-based candidates
        quantum_candidates = []
        vocab = list(user_vocabulary.union(self.base_words))[:50]
        
        for word in vocab:
            word_state = self.semantics.encode_word(word)
            fidelity = float(np.abs(np.sum(np.conj(word_state) * field_state)) ** 2)
            
            # Bonus for user words
            if word in user_vocabulary:
                fidelity *= 1.3
            
            # Bonus for grammatical compatibility with context
            if context:
                phase_align = self.semantics.phase_alignment(context[-1], word)
                fidelity *= (1 + 0.5 * phase_align)
            
            quantum_candidates.append((word, fidelity))
        
        quantum_candidates.sort(key=lambda x: -x[1])
        quantum_candidates = quantum_candidates[:20]
        
        # Combine n-gram and quantum candidates
        if ngram_candidates and quantum_candidates:
            # Hybrid: weight both
            all_candidates = {}
            
            for word, score in ngram_candidates:
                all_candidates[word] = all_candidates.get(word, 0) + 0.7 * score
            
            for word, score in quantum_candidates:
                all_candidates[word] = all_candidates.get(word, 0) + 0.3 * score
            
            # Sort combined
            combined = [(w, s) for w, s in all_candidates.items()]
            combined.sort(key=lambda x: -x[1])
            
            # Probabilistic selection with temperature (scaled by coherence)
            temp = 2.0 + 8.0 * coherence  # Higher coherence = lower temperature (more precise)
            scores = np.array([s for _, s in combined[:10]])
            probs = np.exp(scores * temp) / np.exp(scores * temp).sum()
            
            idx = np.random.choice(len(combined[:10]), p=probs)
            word, score = combined[idx]
            
            return word, score, 'hybrid'
        
        elif ngram_candidates:
            # Pure n-gram
            temp = 1.0 + 5.0 * coherence
            scores = np.array([s for _, s in ngram_candidates[:10]])
            probs = np.exp(scores * temp) / np.exp(scores * temp).sum()
            
            idx = np.random.choice(len(ngram_candidates[:10]), p=probs)
            word, score = ngram_candidates[idx]
            
            return word, score, 'ngram'
        
        else:
            # Pure quantum (fallback)
            temp = 2.0 + 8.0 * coherence
            scores = np.array([s for _, s in quantum_candidates[:10]])
            probs = np.exp(scores * temp) / np.exp(scores * temp).sum()
            
            idx = np.random.choice(len(quantum_candidates[:10]), p=probs)
            word, score = quantum_candidates[idx]
            
            return word, score, 'quantum'
    
    def generate_response(self, field_state: np.ndarray, 
                         user_vocabulary: Set[str],
                         max_words: int = 12,
                         coherence: float = 0.5) -> Dict:
        """
        Generate a complete response.
        """
        words = []
        scores = []
        sources = []
        
        for _ in range(max_words):
            word, score, source = self.generate_word(words, field_state, user_vocabulary, coherence)
            words.append(word)
            scores.append(score)
            sources.append(source)
            
            # Update field state with generated word
            word_state = self.semantics.encode_word(word)
            field_state = 0.7 * field_state + 0.3 * word_state
            norm = np.sqrt(np.sum(np.abs(field_state) ** 2))
            if norm > 1e-10:
                field_state /= norm
        
        # Learn this sequence if coherence is good
        if coherence > 0.3:
            self.ngram_learner.record_sequence(words, coherence)
            if len(words) >= 2:
                self.ngram_learner.mark_used(words[:-1])
        
        response = ' '.join(words)
        
        # Count sources
        source_counts = {
            'ngram': sources.count('ngram'),
            'quantum': sources.count('quantum'),
            'hybrid': sources.count('hybrid')
        }
        
        return {
            'response': response,
            'mean_score': float(np.mean(scores)),
            'coherence': coherence,
            'sources': source_counts,
            'learned_patterns': len(self.ngram_learner.patterns)
        }
