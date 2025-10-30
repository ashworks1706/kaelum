"""Example: Using KaelumAI with ChromaDB for factual verification."""

from kaelum import enhance
from kaelum.core.rag_adapter import ChromaAdapter

# Example 1: Basic ChromaDB setup
try:
    import chromadb
    
    # Initialize ChromaDB
    client = chromadb.Client()
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="knowledge_base",
        metadata={"description": "My knowledge base for fact-checking"}
    )
    
    # Add some knowledge (you'd typically load this from your data)
    collection.add(
        documents=[
            "Paris is the capital of France, established as the capital in 987 AD.",
            "The Eiffel Tower in Paris was completed in 1889 and stands 330 meters tall.",
            "France has a population of approximately 67 million people as of 2023.",
            "Python was created by Guido van Rossum and first released in 1991.",
            "The speed of light in vacuum is exactly 299,792,458 meters per second.",
        ],
        ids=["fact1", "fact2", "fact3", "fact4", "fact5"],
        metadatas=[
            {"category": "geography", "verified": True},
            {"category": "landmarks", "verified": True},
            {"category": "demographics", "verified": True},
            {"category": "technology", "verified": True},
            {"category": "physics", "verified": True},
        ]
    )
    
    # Create adapter
    adapter = ChromaAdapter(collection)
    
    # Example queries with factual verification
    print("=" * 60)
    print("Example 1: Factual Query with Verification")
    print("=" * 60)
    
    result = enhance(
        "What is the capital of France and when was it established?",
        rag_adapter=adapter,
        use_factual_verification=True,
        mode="auto"
    )
    print(result)
    print()
    
    # Example 2: Math query (no RAG needed)
    print("=" * 60)
    print("Example 2: Math Query (RAG not used)")
    print("=" * 60)
    
    result = enhance(
        "What is 15% of 240?",
        rag_adapter=adapter,
        use_factual_verification=False,  # RAG not needed for math
        mode="math"
    )
    print(result)
    print()
    
    # Example 3: Verify potentially incorrect fact
    print("=" * 60)
    print("Example 3: Incorrect Fact Detection")
    print("=" * 60)
    
    result = enhance(
        "The Eiffel Tower was built in 1850",  # Wrong date!
        rag_adapter=adapter,
        use_factual_verification=True,
        mode="auto"
    )
    print(result)
    print()
    
except ImportError:
    print("❌ ChromaDB not installed. Install with: pip install chromadb")
    print("\nExample code:")
    print("""
from kaelum import enhance
from kaelum.core.rag_adapter import ChromaAdapter
import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("my_knowledge")

# Add your knowledge
collection.add(
    documents=["Paris is the capital of France"],
    ids=["fact1"]
)

# Use with KaelumAI
adapter = ChromaAdapter(collection)
result = enhance(
    "What is the capital of France?",
    rag_adapter=adapter,
    use_factual_verification=True
)
print(result)
    """)
