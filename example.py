#!/usr/bin/env python3

from kaelum import enhance, set_reasoning_model

def main():
    print("=" * 70)
    print("Kaelum AI - Reasoning System Example")
    print("=" * 70)
    
    set_reasoning_model(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:3b",
        temperature=0.7,
        enable_routing=True,
        use_symbolic_verification=True,
        max_reflection_iterations=2
    )
    
    queries = [
        "What is the derivative of x^2 + 3x?",
        "Solve: 2x + 6 = 10",
        "All humans are mortal. Socrates is human. Is Socrates mortal?",
        "Calculate 15% tip on $89.90",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Query {i}: {query}")
        print("=" * 70)
        
        result = enhance(query)
        print(result)
        print()

if __name__ == "__main__":
    main()
