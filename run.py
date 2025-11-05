#!/usr/bin/env python3
"""
Kaelum AI - Production Test Runner
Neural routing, LATS search, verification, and reflection system

Directory Structure (.kaelum/):
================================
.kaelum/
├── routing/                    # Neural router learning data
│   ├── model.pt               # Trained router model weights
│   └── training_data.json     # Query outcomes for training
│
├── cache/                      # LATS tree cache (worker results)
│   ├── metadata.json          # Cache index with embeddings
│   └── trees/                 # Serialized LATS trees
│       └── *.json
│
├── active_learning/            # Training data collection
│   └── query_pool.jsonl       # Queries for fine-tuning
│
├── analytics/                  # Performance metrics
│   ├── metrics.jsonl          # Per-inference metrics
│   └── summary.json           # Aggregated statistics
│
├── cache_validation/           # (Optional) Cache quality validation
├── calibration/                # (Optional) Confidence calibration
└── tree_cache/                 # (Legacy) Old cache location - now uses cache/ dir

Notes:
- cache/ is the active cache directory (configurable via --cache-dir)
- tree_cache/ is legacy and should be empty for new installations
- cache_validation/ and calibration/ are optional features (currently unused)
"""

import argparse
import logging
from kaelum import kaelum_enhance_reasoning, set_reasoning_model, get_metrics

def setup_logging(verbose: bool = False):
    """Configure logging for the entire application."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(message)s',
        force=True  # Override any existing configuration
    )
    
    # Silence noisy third-party libraries
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    
    # Set levels for Kaelum loggers (research-focused: show key decisions)
    logging.getLogger("kaelum.orchestrator").setLevel(logging.INFO)
    logging.getLogger("kaelum.router").setLevel(logging.INFO)
    logging.getLogger("kaelum.lats").setLevel(logging.INFO)
    logging.getLogger("kaelum.llm").setLevel(logging.INFO)  # Show LLM calls
    logging.getLogger("kaelum.worker").setLevel(logging.INFO)
    logging.getLogger("kaelum.verification").setLevel(logging.INFO)  # Show verification outcomes
    logging.getLogger("kaelum.reflection").setLevel(logging.INFO)  # Show self-correction
    logging.getLogger("kaelum.cache").setLevel(logging.INFO)  # Show cache hits/misses
    logging.getLogger("kaelum.cache_validator").setLevel(logging.INFO)  # Show LLM validation
    logging.getLogger("kaelum.reward").setLevel(logging.DEBUG if verbose else logging.WARNING)  # Domain scoring (verbose only)

def main():
    parser = argparse.ArgumentParser(
        description="Kaelum AI - Reasoning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a single question
  python run.py "What is 2+2?"
  
  # Run with default test queries
  python run.py
  
  # Custom LLM and embedding model
  python run.py "Solve x^2 + 5x + 6 = 0" --model llama3:8b --embedding-model all-mpnet-base-v2
  
  # Disable routing and force specific worker
  python run.py "Write a Python function to check if a number is prime" --no-routing --worker code
  
  # Adjust LATS search parameters for better accuracy
  python run.py "What is the derivative of x^2?" --max-tree-depth 8 --num-simulations 20
  
  # Enable factual verification and debugging
  python run.py "What is the capital of France?" --enable-factual-verification --debug-verification
        """
    )
    
    # Query argument (optional - if not provided, runs default test queries)
    parser.add_argument("query", nargs="?", default=None,
                       help="Question or task to solve (optional - if not provided, runs default test queries)")
    
    # LLM Configuration
    llm_group = parser.add_argument_group('LLM Configuration')
    llm_group.add_argument("--base-url", default="http://localhost:11434/v1", 
                        help="LLM API base URL (default: http://localhost:11434/v1)")
    llm_group.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="LLM model name (default: Qwen/Qwen2.5-1.5B-Instruct)")
    llm_group.add_argument("--api-key", default=None,
                        help="API key for LLM server (required for vLLM, optional for Ollama)")
    llm_group.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)")
    llm_group.add_argument("--temperature", type=float, default=0.7,
                        help="LLM temperature, higher = more creative (default: 0.7, range: 0.0-2.0)")
    llm_group.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens for LLM response (default: 1024)")
    
    # Routing Configuration
    routing_group = parser.add_argument_group('Routing & Worker Configuration')
    routing_group.add_argument("--no-routing", action="store_true",
                        help="Disable neural router, use default logic worker")
    routing_group.add_argument("--worker", choices=["math", "logic", "code", "factual", "creative", "analysis"],
                        help="Force specific worker (overrides router)")
    routing_group.add_argument("--router-data-dir", default=".kaelum/routing",
                        help="Directory for router learning data (default: .kaelum/routing)")
    
    # LATS Tree Search Configuration
    search_group = parser.add_argument_group('LATS Tree Search Configuration')
    search_group.add_argument("--max-tree-depth", type=int, default=None,
                        help="Max tree search depth (default: router decides 3-10, or 5 if routing disabled)")
    search_group.add_argument("--num-simulations", type=int, default=None,
                        help="Number of LATS simulations (default: router decides 5-25, or 10 if routing disabled)")
    search_group.add_argument("--parallel", action="store_true",
                        help="Enable parallel LATS simulations (faster but uses more resources)")
    search_group.add_argument("--max-workers", type=int, default=4,
                        help="Max parallel workers for LATS (default: 4, requires --parallel)")
    
    # Caching Configuration
    cache_group = parser.add_argument_group('Caching Configuration')
    cache_group.add_argument("--no-cache", action="store_true",
                        help="Disable tree cache (slower but always computes fresh)")
    cache_group.add_argument("--cache-dir", default=".kaelum/cache",
                        help="Directory for tree cache storage (default: .kaelum/cache)")
    
    # Verification Configuration
    verification_group = parser.add_argument_group('Verification Configuration')
    verification_group.add_argument("--no-symbolic-verification", action="store_true",
                        help="Disable symbolic verification with SymPy (for math/logic)")
    verification_group.add_argument("--no-factual-verification", action="store_true",
                        help="Disable factual verification (enabled by default)")
    verification_group.add_argument("--debug-verification", action="store_true",
                        help="Enable detailed verification debugging output")
    
    # Reflection Configuration
    reflection_group = parser.add_argument_group('Reflection & Learning Configuration')
    reflection_group.add_argument("--max-reflection-iterations", type=int, default=2,
                        help="Max self-correction iterations (default: 2, range: 0-5)")
    reflection_group.add_argument("--no-active-learning", action="store_true",
                        help="Disable active learning query collection for fine-tuning")
    
    # Logging Configuration
    logging_group = parser.add_argument_group('Logging Configuration')
    logging_group.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging (shows DEBUG level logs)")
    
    args = parser.parse_args()
    
    # Setup logging based on arguments
    setup_logging(verbose=args.verbose or args.debug_verification)
    
    logger = logging.getLogger("kaelum.run")
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{'Kaelum AI - Reasoning System':^80}")
    logger.info(f"{'=' * 80}")
    
    print("=" * 80)
    print(" " * 22 + "Kaelum AI - Reasoning System")
    print("=" * 80)
    print("\n📋 Configuration:")
    print(f"  LLM: {args.model} @ {args.base_url}")
    if args.api_key:
        print(f"  API Key: {'*' * 8}{args.api_key[-4:] if len(args.api_key) > 4 else '****'}")
    else:
        print(f"  API Key: Not set (may be required for vLLM)")
    print(f"  Embeddings: {args.embedding_model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max Tokens: {args.max_tokens}")
    print(f"\n🧭 Routing & Workers:")
    print(f"  Neural Router: {'✗ Disabled' if args.no_routing else '✓ Enabled'}")
    if args.worker:
        print(f"  Forced Worker: {args.worker}")
    print(f"  Router Data: {args.router_data_dir}")
    print(f"\n🌳 LATS Tree Search:")
    if args.max_tree_depth:
        print(f"  Max Depth: {args.max_tree_depth} (manual override)")
    else:
        print(f"  Max Depth: Router decides (3-10)" if not args.no_routing else f"  Max Depth: 5 (default)")
    if args.num_simulations:
        print(f"  Simulations: {args.num_simulations} (manual override)")
    else:
        print(f"  Simulations: Router decides (5-25)" if not args.no_routing else f"  Simulations: 10 (default)")
    print(f"  Parallel: {'✓ Enabled' if args.parallel else '✗ Disabled'}")
    if args.parallel:
        print(f"  Max Workers: {args.max_workers}")
    print(f"\n💾 Caching:")
    print(f"  Tree Cache: {'✗ Disabled' if args.no_cache else '✓ Enabled'}")
    print(f"  Cache Dir: {args.cache_dir}")
    print(f"\n✓ Verification:")
    print(f"  Symbolic (SymPy): {'✗ Disabled' if args.no_symbolic_verification else '✓ Enabled'}")
    print(f"  Factual: {'✗ Disabled' if args.no_factual_verification else '✓ Enabled'}")
    print(f"  Debug Mode: {'✓ Enabled' if args.debug_verification else '✗ Disabled'}")
    print(f"\n🔄 Reflection & Learning:")
    print(f"  Max Iterations: {args.max_reflection_iterations}")
    print(f"  Active Learning: {'✗ Disabled' if args.no_active_learning else '✓ Enabled'}")
    
    set_reasoning_model(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_routing=not args.no_routing,
        use_symbolic_verification=not args.no_symbolic_verification,
        use_factual_verification=not args.no_factual_verification,
        max_reflection_iterations=args.max_reflection_iterations,
        debug_verification=args.debug_verification,
        enable_active_learning=not args.no_active_learning,
        cache_dir=args.cache_dir,
        router_data_dir=args.router_data_dir,
        parallel=args.parallel,
        max_workers=args.max_workers,
        max_tree_depth=args.max_tree_depth,
        num_simulations=args.num_simulations,
    )
    
    # If a single query is provided, run it and exit
    if args.query:
        print(f"\n{'=' * 80}")
        print(f"Query: {args.query}")
        print("=" * 80)
        
        try:
            result = kaelum_enhance_reasoning(args.query)
            
            print(f"\n{'─' * 80}")
            print(f"ANSWER: {result.get('suggested_approach', 'N/A')}")
            print(f"{'─' * 80}")
            
            print(f"\nWorker: {result.get('worker_used', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            print(f"Verification: {'✓ PASSED' if result.get('verification_passed') else '✗ FAILED'}")
            print(f"Cache Hit: {'Yes' if result.get('cache_hit') else 'No'}")
            print(f"Iterations: {result.get('iterations', 0)}")
            
            print(f"\nReasoning Steps:")
            print(f"  Step Count: {result.get('reasoning_count', 0)}")
            
            reasoning = result.get('reasoning_steps', [])
            if reasoning:
                for j, step in enumerate(reasoning, 1):
                    print(f"  {j}. {step}")
            
            # Display metrics
            print(f"\n{'─' * 80}")
            print("METRICS:")
            print(f"{'─' * 80}")
            
            try:
                metrics = get_metrics()
                
                # Session metrics
                if 'session' in metrics and metrics['session']:
                    session = metrics['session']
                    print(f"\n📊 Session Stats:")
                    print(f"  Total Inferences: {session.get('total_inferences', 0)}")
                    print(f"  Cache Hits: {session.get('cache_hits', 0)}")
                    print(f"  Cache Hit Rate: {session.get('cache_hit_rate', 0)*100:.1f}%")
                    print(f"  Avg Latency: {session.get('avg_latency_ms', 0):.0f}ms")
                    
                    if session.get('total_input_tokens', 0) > 0:
                        print(f"\n🔤 Token Usage:")
                        print(f"  Input Tokens: {session.get('total_input_tokens', 0):,}")
                        print(f"  Output Tokens: {session.get('total_output_tokens', 0):,}")
                        print(f"  Total Tokens: {session.get('total_tokens', 0):,}")
                
                # Analytics
                if 'analytics' in metrics and metrics['analytics']:
                    analytics = metrics['analytics']
                    
                    if analytics.get('worker_distribution'):
                        print(f"\n👷 Worker Distribution:")
                        for worker, count in analytics['worker_distribution'].items():
                            print(f"  {worker}: {count}")
                    
                    if analytics.get('verification_rate') is not None:
                        print(f"\n✓ Verification:")
                        print(f"  Pass Rate: {analytics.get('verification_rate', 0)*100:.1f}%")
                    
                    if analytics.get('avg_simulations') is not None:
                        print(f"\n🌳 LATS Stats:")
                        print(f"  Avg Simulations: {analytics.get('avg_simulations', 0):.1f}")
                        print(f"  Avg Tree Depth: {analytics.get('avg_tree_depth', 0):.1f}")
            
            except Exception as e:
                print(f"  Could not load metrics: {e}")
            
            print()
            return
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Otherwise, run default test queries
    queries = [
        ("Math", "What is the derivative of x^2 + 3x?"),
        ("Math", "Solve: 2x + 6 = 10"),
        ("Logic", "All humans are mortal. Socrates is human. Is Socrates mortal?"),
        ("Math", "Calculate 15% tip on $89.90"),
        ("Code", "Write a Python function to check if a number is prime"),
        ("Factual", "What is the capital of France and its population?"),
    ]
    
    results_summary = []
    
    for i, (domain, query) in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}/{len(queries)} [{domain}]: {query}")
        print("=" * 80)
        
        try:
            result = kaelum_enhance_reasoning(query)
            
            print(f"\n{'─' * 80}")
            print(f"ANSWER: {result.get('suggested_approach', 'N/A')}")
            print(f"{'─' * 80}")
            
            print(f"\nWorker: {result.get('worker_used', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            print(f"Verification: {'✓ PASSED' if result.get('verification_passed') else '✗ FAILED'}")
            print(f"Cache Hit: {'Yes' if result.get('cache_hit') else 'No'}")
            print(f"Iterations: {result.get('iterations', 0)}")
            
            print(f"\nReasoning Steps:")
            print(f"  Step Count: {result.get('reasoning_count', 0)}")
            
            reasoning = result.get('reasoning_steps', [])
            if reasoning:
                for j, step in enumerate(reasoning[:5], 1):
                    print(f"  {j}. {step}")
                if len(reasoning) > 5:
                    print(f"  ... ({len(reasoning) - 5} more steps)")
            
            results_summary.append({
                'domain': domain,
                'query': query[:50] + '...' if len(query) > 50 else query,
                'worker': result.get('worker_used', 'N/A'),
                'verified': result.get('verification_passed', False),
                'time_ms': 0  # metrics not available in simplified response
            })
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            results_summary.append({
                'domain': domain,
                'query': query[:50] + '...' if len(query) > 50 else query,
                'worker': 'ERROR',
                'verified': False,
                'time_ms': 0
            })
    
    print(f"\n{'=' * 80}")
    print(" " * 30 + "SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Domain':<10} {'Worker':<10} {'Verified':<10} {'Time (ms)':<12} {'Query'}")
    print("─" * 80)
    
    for r in results_summary:
        verified_icon = '✓' if r['verified'] else '✗'
        print(f"{r['domain']:<10} {r['worker']:<10} {verified_icon:<10} {r['time_ms']:<12.0f} {r['query']}")
    
    total_verified = sum(1 for r in results_summary if r['verified'])
    avg_time = sum(r['time_ms'] for r in results_summary) / len(results_summary)
    
    print("─" * 80)
    print(f"\nSuccess Rate: {total_verified}/{len(results_summary)} ({total_verified/len(results_summary)*100:.1f}%)")
    print(f"Average Time: {avg_time:.0f}ms")
    print(f"\n✓ Neural router trained on {len(results_summary)} outcomes")
    print(f"✓ Model saved to .kaelum/routing/model.pt")
    print(f"✓ Run again to see improved routing and cache hits!\n")

if __name__ == "__main__":
    main()
