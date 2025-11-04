#!/usr/bin/env python
"""Example usage of the Neural Router (Kaelum Brain).

This script demonstrates:
1. Training the neural router on synthetic data
2. Using the neural router for query routing
3. Comparing neural vs rule-based routing
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from kaelum.core.neural_router import NeuralRouter
    from kaelum.core.neural_router_trainer import NeuralRouterTrainer
    from kaelum.core.router import Router
    NEURAL_ROUTER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Neural router not available: {e}")
    print("   Install dependencies: pip install torch sentence-transformers")
    sys.exit(1)


def train_neural_router():
    """Train the neural router with synthetic data."""
    print("\n" + "=" * 70)
    print("🧠 TRAINING NEURAL ROUTER (KAELUM BRAIN)")
    print("=" * 70)
    
    # Initialize neural router
    neural_router = NeuralRouter(
        data_dir=".kaelum/neural_routing",
        fallback_to_rules=True,
        device="cpu"
    )
    
    # Initialize trainer
    trainer = NeuralRouterTrainer(
        neural_router=neural_router,
        learning_rate=0.001,
        batch_size=32,
        device="cpu"
    )
    
    # Generate synthetic training data
    print("\n📝 Generating synthetic training data...")
    trainer.generate_synthetic_data(num_samples=500, save_to_outcomes=True)
    
    # Train the model
    print("\n🚀 Training neural router...")
    history = trainer.train(
        num_epochs=30,
        validation_split=0.2,
        early_stopping_patience=5,
        save_best=True
    )
    
    print("\n✅ Training complete!")
    print(f"   Final validation accuracy: {history['val_accuracy'][-1]:.2%}")
    print(f"   Model saved to: .kaelum/neural_routing/neural_router.pt")
    
    return neural_router


def test_routing_comparison():
    """Compare neural router vs rule-based router."""
    print("\n" + "=" * 70)
    print("🔬 COMPARING NEURAL VS RULE-BASED ROUTING")
    print("=" * 70)
    
    # Test queries
    test_queries = [
        "Calculate 15% of 250",
        "Solve the equation: 3x + 7 = 22",
        "Write a Python function to reverse a list",
        "Who was the first person to walk on the moon?",
        "Write a haiku about artificial intelligence",
        "Compare the advantages and disadvantages of renewable energy",
        "If all cats are mammals and all mammals are animals, are all cats animals?",
        "Implement a binary search algorithm in Python",
    ]
    
    # Initialize both routers
    print("\n📦 Initializing routers...")
    neural_router = NeuralRouter(
        data_dir=".kaelum/neural_routing",
        fallback_to_rules=True,
        device="cpu"
    )
    
    rule_router = Router(learning_enabled=False)
    
    # Test each query
    print("\n" + "-" * 70)
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print()
        
        # Neural routing
        start = time.time()
        neural_decision = neural_router.route(query)
        neural_time = (time.time() - start) * 1000
        
        # Rule-based routing
        start = time.time()
        rule_decision = rule_router.route(query)
        rule_time = (time.time() - start) * 1000
        
        # Compare results
        print(f"   Neural Router:")
        print(f"      Strategy: {neural_decision.strategy.value}")
        print(f"      Reflection: {neural_decision.max_reflection_iterations}")
        print(f"      Symbolic: {neural_decision.use_symbolic_verification}")
        print(f"      Factual: {neural_decision.use_factual_verification}")
        print(f"      Time: {neural_time:.2f}ms")
        
        print(f"\n   Rule-Based Router:")
        print(f"      Strategy: {rule_decision.strategy.value}")
        print(f"      Reflection: {rule_decision.max_reflection_iterations}")
        print(f"      Symbolic: {rule_decision.use_symbolic_verification}")
        print(f"      Factual: {rule_decision.use_factual_verification}")
        print(f"      Time: {rule_time:.2f}ms")
        
        # Highlight differences
        if neural_decision.strategy != rule_decision.strategy:
            print(f"\n   ⚠️  Strategy differs: Neural={neural_decision.strategy.value}, Rule={rule_decision.strategy.value}")
        else:
            print(f"\n   ✓ Strategy matches: {neural_decision.strategy.value}")
        
        print("-" * 70)


def interactive_routing():
    """Interactive mode for testing routing."""
    print("\n" + "=" * 70)
    print("🎮 INTERACTIVE NEURAL ROUTER")
    print("=" * 70)
    print("\nEnter queries to see routing decisions (type 'quit' to exit)")
    
    # Initialize neural router
    neural_router = NeuralRouter(
        data_dir=".kaelum/neural_routing",
        fallback_to_rules=True,
        device="cpu"
    )
    
    while True:
        print("\n" + "-" * 70)
        query = input("\n🔍 Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Get routing decision
        decision = neural_router.route(query)
        
        # Display results
        print("\n📊 Routing Decision:")
        print(f"   Query Type: {decision.query_type.value}")
        print(f"   Strategy: {decision.strategy.value}")
        print(f"   Max Reflection: {decision.max_reflection_iterations}")
        print(f"   Symbolic Verification: {decision.use_symbolic_verification}")
        print(f"   Factual Verification: {decision.use_factual_verification}")
        print(f"   Confidence Threshold: {decision.confidence_threshold:.2f}")
        print(f"   Complexity: {decision.complexity_score:.2f}")
        print(f"   Reasoning: {decision.reasoning}")


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("🧠 NEURAL ROUTER (KAELUM BRAIN) DEMO")
    print("=" * 70)
    print("\nThis demo will:")
    print("1. Train the neural router with synthetic data")
    print("2. Compare neural vs rule-based routing on test queries")
    print("3. Start interactive mode for testing")
    print()
    
    # Check if model already exists
    model_path = Path(".kaelum/neural_routing/neural_router.pt")
    
    if model_path.exists():
        print(f"✓ Found existing model at {model_path}")
        response = input("   Train a new model? (y/N): ").strip().lower()
        if response == 'y':
            train_neural_router()
    else:
        print("⚠️  No existing model found - training required")
        train_neural_router()
    
    # Run comparison
    test_routing_comparison()
    
    # Interactive mode
    print("\n" + "=" * 70)
    response = input("\n🎮 Start interactive mode? (Y/n): ").strip().lower()
    if response != 'n':
        interactive_routing()
    
    print("\n" + "=" * 70)
    print("👋 Demo complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
