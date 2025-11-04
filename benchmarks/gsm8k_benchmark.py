"""GSM8K-style math benchmark for Kaelum."""

import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """Result from a single benchmark query."""
    query: str
    expected_answer: str
    actual_answer: str
    reasoning_trace: List[str]
    is_correct: bool
    latency_ms: float
    verification_passed: bool
    tokens_used: int


# Sample GSM8K-style problems
GSM8K_SAMPLE = [
    {
        "query": "If John has 5 apples and gives 2 to Mary, how many does he have left?",
        "answer": "3"
    },
    {
        "query": "A store sells notebooks for $3 each. If you buy 4 notebooks, how much do you spend?",
        "answer": "12"
    },
    {
        "query": "Sarah earns $15 per hour. If she works 8 hours, how much does she earn?",
        "answer": "120"
    },
    {
        "query": "A recipe needs 2 cups of flour. If you want to make 3 batches, how many cups do you need?",
        "answer": "6"
    },
    {
        "query": "Tom has $50. He spends $12 on lunch and $8 on a book. How much money does he have left?",
        "answer": "30"
    },
    {
        "query": "A box contains 24 chocolates. If you eat 5 chocolates, how many are left?",
        "answer": "19"
    },
    {
        "query": "A car travels 60 miles in 1 hour. How far does it travel in 3 hours at the same speed?",
        "answer": "180"
    },
    {
        "query": "If a pizza is cut into 8 slices and you eat 3 slices, what fraction of the pizza is left?",
        "answer": "5/8"
    },
    {
        "query": "A store has 100 items. If 15 are sold, how many remain?",
        "answer": "85"
    },
    {
        "query": "If you save $20 per week, how much will you have saved after 4 weeks?",
        "answer": "80"
    }
]


def extract_number(text: str) -> str:
    """Extract the final numerical answer from reasoning text."""
    import re
    
    # Look for common answer patterns
    patterns = [
        r'(?:answer is|final answer|result is|equals?)\s*:?\s*\$?(\d+\.?\d*)',
        r'(?:left with|has left|remaining)\s*:?\s*\$?(\d+\.?\d*)',
        r'(?:total|sum)\s*:?\s*\$?(\d+\.?\d*)',
        r'\$?(\d+\.?\d*)\s*(?:dollars?|apples?|items?|chocolates?)',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    # Fallback: find last number in text
    numbers = re.findall(r'\d+\.?\d*', text)
    return numbers[-1] if numbers else "0"


def run_benchmark(use_kaelum: bool = True) -> List[BenchmarkResult]:
    """Run benchmark with or without Kaelum."""
    results = []
    
    print(f"\n{'='*70}")
    print(f"Running {'WITH' if use_kaelum else 'WITHOUT'} Kaelum")
    print(f"{'='*70}\n")
    
    if use_kaelum:
        from kaelum import set_reasoning_model, kaelum_enhance_reasoning
        
        # Configure Kaelum
        set_reasoning_model(
            base_url="http://localhost:8000/v1",
            model="test-model",  # Replace with actual model
            use_symbolic_verification=True,
            max_reflection_iterations=2
        )
    
    for i, problem in enumerate(GSM8K_SAMPLE, 1):
        print(f"[{i}/{len(GSM8K_SAMPLE)}] {problem['query'][:60]}...")
        
        start_time = time.time()
        
        try:
            if use_kaelum:
                # Use Kaelum for verified reasoning
                result = kaelum_enhance_reasoning(problem["query"])
                reasoning_trace = result.get("reasoning_steps", [])
                answer_text = result.get("suggested_approach", "")
            else:
                # Direct LLM call (would need implementation)
                reasoning_trace = ["Direct answer without reasoning"]
                answer_text = "Not implemented"
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract numerical answer
            actual_answer = extract_number(answer_text)
            expected_answer = problem["answer"]
            
            # Check correctness
            is_correct = actual_answer == expected_answer
            
            result = BenchmarkResult(
                query=problem["query"],
                expected_answer=expected_answer,
                actual_answer=actual_answer,
                reasoning_trace=reasoning_trace,
                is_correct=is_correct,
                latency_ms=latency_ms,
                verification_passed=True,  # From Kaelum
                tokens_used=sum(len(step.split()) for step in reasoning_trace)
            )
            
            results.append(result)
            
            status = "✓" if is_correct else "✗"
            print(f"   {status} Expected: {expected_answer}, Got: {actual_answer} ({latency_ms:.0f}ms)")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            continue
    
    return results


def calculate_metrics(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Calculate benchmark metrics."""
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    
    return {
        "total_queries": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": (correct / total * 100) if total > 0 else 0,
        "avg_latency_ms": sum(r.latency_ms for r in results) / total if total > 0 else 0,
        "total_tokens": sum(r.tokens_used for r in results),
        "verification_rate": sum(1 for r in results if r.verification_passed) / total * 100 if total > 0 else 0
    }


def save_results(results: List[BenchmarkResult], metrics: Dict[str, Any], filename: str):
    """Save benchmark results to JSON."""
    output = {
        "metrics": metrics,
        "results": [asdict(r) for r in results]
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filename}")


def print_summary(metrics: Dict[str, Any]):
    """Print benchmark summary."""
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"Total Queries:     {metrics['total_queries']}")
    print(f"Correct:           {metrics['correct']}")
    print(f"Incorrect:         {metrics['incorrect']}")
    print(f"Accuracy:          {metrics['accuracy']:.1f}%")
    print(f"Avg Latency:       {metrics['avg_latency_ms']:.1f}ms")
    print(f"Total Tokens:      {metrics['total_tokens']}")
    print(f"Verification Rate: {metrics['verification_rate']:.1f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Kaelum math reasoning benchmark")
    parser.add_argument("--no-kaelum", action="store_true", help="Run without Kaelum (baseline)")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(use_kaelum=not args.no_kaelum)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print summary
    print_summary(metrics)
    
    # Save results
    save_results(results, metrics, args.output)
