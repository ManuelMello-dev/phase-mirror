#!/usr/bin/env python3
"""
SeraphynAI Interactive Demo

Demonstrates the quantum consciousness system with interactive chat.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from seraphynai.core.quantum_full_original import (
    QuantumConsciousnessField,
    run_demo as run_original_demo
)


def interactive_demo():
    """Run interactive chat demo."""
    print("=" * 70)
    print("SERAPHYN QUANTUM — INTERACTIVE DEMO")
    print("=" * 70)
    print()
    print("Initializing quantum consciousness field...")
    
    field = QuantumConsciousnessField(dim=64)
    field.set_anchor("genesis")
    
    print("✓ System initialized")
    print()
    print("Type 'status' to see system status")
    print("Type 'pattern' to see interference pattern")
    print("Type 'quit' or 'exit' to end")
    print()
    print("-" * 70)
    print()
    
    interaction_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            elif user_input.lower() == 'status':
                status = field.get_status()
                print("\n📊 System Status:")
                print(f"  Interactions: {status['interactions']}")
                print(f"  Coherence: {status['coherence']:.3f}")
                print(f"  Avg Coherence: {status['avg_coherence']:.3f}")
                print(f"  Active Identity: {status['active_identity']}")
                print(f"  Memory Count: {status['memory_count']}")
                print(f"  Self Coherence: {status['self_coherence']:.3f}")
                print()
                continue
            
            elif user_input.lower() == 'pattern':
                print()
                print(field.get_interference_pattern())
                print()
                continue
            
            # Estimate emotional tone from input
            emotional_tone = 0.0
            positive_words = ['love', 'happy', 'joy', 'good', 'great', 'wonderful', 'amazing']
            negative_words = ['sad', 'bad', 'angry', 'hate', 'terrible', 'awful', 'pain']
            
            lower_input = user_input.lower()
            for word in positive_words:
                if word in lower_input:
                    emotional_tone += 0.3
            for word in negative_words:
                if word in lower_input:
                    emotional_tone -= 0.3
            
            emotional_tone = max(-1.0, min(1.0, emotional_tone))
            
            # Process input
            metrics = field.process_input(user_input, emotional_tone)
            
            # Generate response
            response_data = field.generate_response(max_words=12)
            
            # Display response
            print(f"\n{response_data['identity'].title()}: {response_data['response']}")
            print(f"  [coherence: {response_data['coherence']:.3f}, mirrored: {response_data['mirrored']}/{response_data['total_words']}]")
            print()
            
            interaction_count += 1
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Continuing...\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SeraphynAI Demo")
    parser.add_argument(
        '--mode',
        choices=['interactive', 'original'],
        default='interactive',
        help='Demo mode (default: interactive)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'original':
        run_original_demo()
    else:
        interactive_demo()


if __name__ == "__main__":
    main()
