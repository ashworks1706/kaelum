#!/usr/bin/env python3
"""KaelumAI CLI - Quick reasoning from the command line."""

import sys
from typing import Optional

import argparse

from kaelum import enhance


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="KaelumAI - Make any LLM reason better",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kaelum "What is 15% of 240?"
  kaelum "Solve x^2 + 5x + 6 = 0" --mode math
  kaelum "Explain how binary search works" --mode code --stream
  kaelum "Is it ethical to lie to save a life?" --mode logic
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="Your question or problem"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "math", "code", "logic", "creative"],
        default="auto",
        help="Reasoning mode (default: auto)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (auto-detects Ollama if not specified)"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream reasoning steps in real-time"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Max reflection cycles (default: 1)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    
    parser.add_argument(
        "--api-base",
        type=str,
        help="Custom API endpoint"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="KaelumAI 0.2.0"
    )
    
    args = parser.parse_args()
    
    try:
        # Run enhancement
        result = enhance(
            query=args.query,
            mode=args.mode,
            model=args.model,
            stream=args.stream,
            max_iterations=args.max_iterations,
            cache=not args.no_cache,
            api_base=args.api_base,
            api_key=args.api_key,
        )
        
        # Output
        if args.stream:
            for chunk in result:
                print(chunk, end="", flush=True)
            print()  # Final newline
        else:
            print(result)
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user", file=sys.stderr)
        return 130
    
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("\nTroubleshooting:", file=sys.stderr)
        print("  • Check if Ollama is running: ollama list", file=sys.stderr)
        print("  • Try specifying a model: kaelum 'your query' --model qwen2.5:7b", file=sys.stderr)
        print("  • Check API connection: curl http://localhost:11434/api/tags", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
