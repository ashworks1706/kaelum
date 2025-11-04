"""Test script for the hybrid math processing approach."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "kaelum"))

from kaelum import set_reasoning_model, enhance

# Test cases for math processing
test_cases = [
    # Basic algebra
    "Solve the equation x^2 - 4 = 0",
    
    # Calculus
    "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
    
    # Integration
    "Calculate the integral of sin(x) from 0 to π",
    
    # Multivariate calculus
    "Find the partial derivative of f(x,y) = x^2*y + sin(x*y) with respect to x",
    
    # Mixed problem
    "If f(x) = x^2 + 3x, find f'(2) and verify that the derivative at x=2 equals 7"
]

def test_hybrid_approach():
    """Test the hybrid approach with math standardization."""
    print("Testing Hybrid Math Processing Approach")
    print("=" * 50)
    
    # Initialize with strict math format enabled
    set_reasoning_model(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b",
        debug_verification=True,
        strict_math_format=True
    )
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case}")
        print("-" * 30)
        
        try:
            result = enhance(test_case)
            print("Result:")
            print(result[:500] + "..." if len(result) > 500 else result)
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()

if __name__ == "__main__":
    test_hybrid_approach()