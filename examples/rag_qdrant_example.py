"""Example: Using KaelumAI with Qdrant for factual verification."""

from kaelum import enhance
from kaelum.core.rag_adapter import QdrantAdapter

# Example: Qdrant setup
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
    
    # Initialize Qdrant (in-memory for demo)
    client = QdrantClient(":memory:")
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create collection
    collection_name = "knowledge_base"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # Add knowledge
    documents = [
        "Paris is the capital of France, established as the capital in 987 AD.",
        "The Eiffel Tower in Paris was completed in 1889 and stands 330 meters tall.",
        "Python was created by Guido van Rossum and first released in 1991.",
    ]
    
    # Embed and upload
    for i, doc in enumerate(documents):
        vector = embedding_model.encode(doc).tolist()
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={"text": doc, "category": "knowledge"}
                )
            ]
        )
    
    # Create adapter
    def embed_function(text):
        return embedding_model.encode(text).tolist()
    
    adapter = QdrantAdapter(
        client=client,
        collection_name=collection_name,
        embedding_function=embed_function
    )
    
    # Use with KaelumAI
    print("=" * 60)
    print("Example: Qdrant Factual Verification")
    print("=" * 60)
    
    result = enhance(
        "When was the Eiffel Tower built?",
        rag_adapter=adapter,
        use_factual_verification=True,
        mode="auto"
    )
    print(result)
    
except ImportError as e:
    print(f"❌ Dependencies not installed: {e}")
    print("\nInstall with:")
    print("  pip install qdrant-client sentence-transformers")
    print("\nExample code:")
    print("""
from kaelum import enhance
from kaelum.core.rag_adapter import QdrantAdapter
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(":memory:")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Setup collection and add documents...

adapter = QdrantAdapter(
    client=client,
    collection_name="my_collection",
    embedding_function=lambda x: embedding_model.encode(x).tolist()
)

result = enhance(
    "Your query",
    rag_adapter=adapter,
    use_factual_verification=True
)
    """)
