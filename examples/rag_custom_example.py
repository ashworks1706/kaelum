"""Example: Using KaelumAI with custom RAG adapter."""

from kaelum import enhance
from kaelum.core.rag_adapter import CustomRAGAdapter

# Example: Custom RAG implementation
print("=" * 60)
print("Example: Custom RAG Adapter")
print("=" * 60)

# Your custom knowledge base (could be any data structure)
knowledge_base = {
    "facts": [
        {"text": "Water boils at 100°C at sea level", "category": "science"},
        {"text": "The Earth orbits the Sun once per year", "category": "astronomy"},
        {"text": "Python is a programming language", "category": "technology"},
    ]
}

# Define custom search function
def my_search_function(query: str, top_k: int = 5):
    """Custom search implementation."""
    results = []
    query_lower = query.lower()
    
    for fact in knowledge_base["facts"]:
        # Simple keyword matching (you'd use embeddings in real implementation)
        score = sum(1 for word in query_lower.split() if word in fact["text"].lower())
        
        if score > 0:
            results.append({
                "text": fact["text"],
                "score": score / len(query_lower.split()),  # Normalize
                "metadata": {"category": fact["category"]}
            })
    
    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# Optional: Define custom verify function
def my_verify_function(claim: str, context=None):
    """Custom verification logic."""
    results = my_search_function(claim, top_k=1)
    
    if not results:
        return True, 0.5  # Neutral if no data
    
    # Check if claim matches knowledge base
    best_match = results[0]
    confidence = best_match["score"]
    is_verified = confidence > 0.5
    
    return is_verified, confidence

# Create custom adapter
adapter = CustomRAGAdapter(
    search_function=my_search_function,
    verify_function=my_verify_function  # Optional
)

# Test 1: Query that matches knowledge base
print("\n1. Query matching knowledge base:")
result = enhance(
    "What temperature does water boil at?",
    rag_adapter=adapter,
    use_factual_verification=True,
    mode="auto"
)
print(result)
print()

# Test 2: Query not in knowledge base
print("\n2. Query not in knowledge base:")
result = enhance(
    "What is the capital of France?",
    rag_adapter=adapter,
    use_factual_verification=True,
    mode="auto"
)
print(result)
print()

# Test 3: Math query (RAG not used)
print("\n3. Math query (RAG ignored):")
result = enhance(
    "What is 25% of 80?",
    rag_adapter=adapter,
    use_factual_verification=False,  # Don't use RAG for math
    mode="math"
)
print(result)

print("\n" + "=" * 60)
print("✅ Custom RAG adapter works!")
print("=" * 60)
