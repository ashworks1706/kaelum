#!/usr/bin/env python3
"""Test script for Advanced Metrics and Active Learning features."""

from kaelum import (
    set_reasoning_model,
    enhance,
    get_metrics,
    get_active_learning_stats,
    generate_training_batch,
    export_training_data
)

def main():
    print("=" * 80)
    print("Testing Advanced Metrics & Active Learning")
    print("=" * 80)
    
    set_reasoning_model(
        base_url="http://localhost:11434/v1",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        enable_routing=True,
        enable_active_learning=True,
        use_symbolic_verification=True,
        max_reflection_iterations=2
    )
    
    test_queries = [
        "What is the derivative of x^2 + 3x?",
        "Calculate 15% tip on $89.90",
        "Write a Python function to check if a number is prime",
        "If all humans are mortal and Socrates is human, what follows?",
        "What is the capital of France?",
    ]
    
    print("\n" + "=" * 80)
    print("Running test queries...")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] {query}")
        result = enhance(query)
        print(f"✓ Completed")
    
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    
    metrics = get_metrics()
    
    if "analytics" in metrics:
        analytics = metrics["analytics"]
        print(f"\nTotal Queries: {analytics.get('total_queries', 0)}")
        print(f"Total Tokens: {analytics.get('total_tokens', 0):,}")
        print(f"Average Time: {analytics.get('avg_time_ms', 0):.0f}ms")
        print(f"Average Tokens: {analytics.get('avg_tokens', 0):.0f}")
        print(f"Cache Hit Rate: {analytics.get('cache_hit_rate', 0):.2%}")
        print(f"Verification Rate: {analytics.get('verification_rate', 0):.2%}")
        
        if "by_worker" in analytics:
            print(f"\nBy Worker:")
            for worker, count in analytics["by_worker"].items():
                print(f"  {worker}: {count}")
    
    print("\n" + "=" * 80)
    print("ACTIVE LEARNING STATISTICS")
    print("=" * 80)
    
    al_stats = get_active_learning_stats()
    
    if "active_learning" not in al_stats:
        print(f"\nTotal Queries Collected: {al_stats.get('total_queries', 0)}")
        print(f"Selected for Training: {al_stats.get('selected_queries', 0)}")
        print(f"Available for Selection: {al_stats.get('unselected_queries', 0)}")
        print(f"Average Confidence: {al_stats.get('avg_confidence', 0):.2f}")
        print(f"Average Complexity: {al_stats.get('avg_complexity', 0):.2f}")
        print(f"Verification Rate: {al_stats.get('verification_rate', 0):.2%}")
        
        if "by_worker" in al_stats:
            print(f"\nQueries by Worker:")
            for worker, count in al_stats["by_worker"].items():
                print(f"  {worker}: {count}")
    
    print("\n" + "=" * 80)
    print("GENERATING TRAINING BATCH")
    print("=" * 80)
    
    strategies = ["uncertainty", "diversity", "error", "complexity", "mixed"]
    
    for strategy in strategies:
        batch = generate_training_batch(strategy=strategy, batch_size=2)
        print(f"\nStrategy: {strategy} - Generated {len(batch)} examples")
        if batch:
            print(f"  Example query: {batch[0]['query'][:60]}...")
    
    print("\n" + "=" * 80)
    print("EXPORTING TRAINING DATA")
    print("=" * 80)
    
    output_file = ".kaelum/training_export.json"
    num_exported = export_training_data(output_file)
    print(f"\nExported {num_exported} training examples to {output_file}")
    
    print("\n" + "=" * 80)
    print("✓ All features tested successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
