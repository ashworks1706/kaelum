#!/usr/bin/env python3
"""Export cache validation training data for fine-tuning"""

import argparse
import json
from pathlib import Path
from core.cache_validator import CacheValidator


def main():
    parser = argparse.ArgumentParser(description='Export cache validation data for fine-tuning')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output file path (default: .kaelum/cache_validation/training_data_<timestamp>.jsonl)')
    parser.add_argument('--stats', action='store_true',
                       help='Show validation statistics')
    
    args = parser.parse_args()
    
    validator = CacheValidator()
    
    if args.stats:
        stats = validator.get_validation_stats()
        print("\n" + "=" * 70)
        print("Cache Validation Statistics")
        print("=" * 70)
        print(f"Total Validations: {stats['total']}")
        print(f"Valid (can reuse):  {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "Valid: 0")
        print(f"Invalid (rejected): {stats['invalid']} ({stats['invalid']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "Invalid: 0")
        print(f"Avg Confidence:    {stats['avg_confidence']:.3f}")
        print(f"Rejection Rate:    {stats['rejection_rate']*100:.1f}%")
        print("=" * 70)
        print()
    
    output_file = validator.export_training_data(args.output)
    
    if Path(output_file).exists():
        with open(output_file, 'r') as f:
            count = sum(1 for _ in f)
        
        print(f"✅ Exported {count} training examples to: {output_file}")
        print()
        print("Use this data to fine-tune your LLM:")
        print(f"  1. Review the data: cat {output_file} | head -n 5")
        print(f"  2. Fine-tune with your framework (HuggingFace, OpenAI, etc.)")
        print(f"  3. Deploy fine-tuned model for better cache validation")
        print()
    else:
        print("❌ No validation data available yet")
        print("   Cache validation data will be collected as queries are processed")


if __name__ == "__main__":
    main()
