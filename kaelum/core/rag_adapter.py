"""RAG adapter for pluggable vector database support."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class RAGAdapter(ABC):
    """Abstract base class for RAG database adapters."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector database for relevant documents.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of documents with metadata
            [{"text": "...", "score": 0.95, "metadata": {...}}, ...]
        """
        pass

    @abstractmethod
    def verify_claim(self, claim: str, context: Optional[List[str]] = None) -> Tuple[bool, float]:
        """
        Verify a factual claim against the knowledge base.

        Args:
            claim: The claim to verify
            context: Optional reasoning context

        Returns:
            (is_verified, confidence_score)
        """
        pass


class ChromaAdapter(RAGAdapter):
    """Adapter for ChromaDB."""

    def __init__(self, collection, embedding_function=None):
        """
        Initialize Chroma adapter.

        Args:
            collection: ChromaDB collection instance
            embedding_function: Optional custom embedding function
        """
        self.collection = collection
        self.embedding_function = embedding_function

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search ChromaDB collection."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        documents = []
        if results and 'documents' in results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    'text': doc,
                    'score': 1 - results['distances'][0][i] if 'distances' in results else 0.9,
                    'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                })

        return documents

    def verify_claim(self, claim: str, context: Optional[List[str]] = None) -> Tuple[bool, float]:
        """Verify claim against ChromaDB."""
        results = self.search(claim, top_k=3)
        
        if not results:
            return True, 0.5  # No data, neutral confidence

        # Calculate average similarity
        avg_score = sum(r['score'] for r in results) / len(results)
        
        # High similarity = claim supported by knowledge base
        is_verified = avg_score > 0.7
        confidence = avg_score
        
        return is_verified, confidence


class QdrantAdapter(RAGAdapter):
    """Adapter for Qdrant."""

    def __init__(self, client, collection_name: str, embedding_function=None):
        """
        Initialize Qdrant adapter.

        Args:
            client: Qdrant client instance
            collection_name: Name of the collection
            embedding_function: Function to convert text to embeddings
        """
        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search Qdrant collection."""
        if not self.embedding_function:
            raise ValueError("embedding_function required for Qdrant")

        # Generate query embedding
        query_vector = self.embedding_function(query)

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        documents = []
        for result in results:
            documents.append({
                'text': result.payload.get('text', ''),
                'score': result.score,
                'metadata': result.payload
            })

        return documents

    def verify_claim(self, claim: str, context: Optional[List[str]] = None) -> Tuple[bool, float]:
        """Verify claim against Qdrant."""
        results = self.search(claim, top_k=3)
        
        if not results:
            return True, 0.5

        avg_score = sum(r['score'] for r in results) / len(results)
        is_verified = avg_score > 0.7
        confidence = avg_score
        
        return is_verified, confidence


class WeaviateAdapter(RAGAdapter):
    """Adapter for Weaviate."""

    def __init__(self, client, class_name: str):
        """
        Initialize Weaviate adapter.

        Args:
            client: Weaviate client instance
            class_name: Name of the Weaviate class
        """
        self.client = client
        self.class_name = class_name

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search Weaviate collection."""
        result = (
            self.client.query
            .get(self.class_name, ["text", "metadata"])
            .with_near_text({"concepts": [query]})
            .with_limit(top_k)
            .with_additional(["distance", "certainty"])
            .do()
        )

        documents = []
        if result and 'data' in result and 'Get' in result['data']:
            for item in result['data']['Get'].get(self.class_name, []):
                documents.append({
                    'text': item.get('text', ''),
                    'score': item['_additional'].get('certainty', 0.9),
                    'metadata': item.get('metadata', {})
                })

        return documents

    def verify_claim(self, claim: str, context: Optional[List[str]] = None) -> Tuple[bool, float]:
        """Verify claim against Weaviate."""
        results = self.search(claim, top_k=3)
        
        if not results:
            return True, 0.5

        avg_score = sum(r['score'] for r in results) / len(results)
        is_verified = avg_score > 0.7
        confidence = avg_score
        
        return is_verified, confidence


class CustomRAGAdapter(RAGAdapter):
    """
    Adapter for custom RAG implementations.
    
    Pass your own search and verify functions.
    """

    def __init__(
        self,
        search_function,
        verify_function=None
    ):
        """
        Initialize custom adapter.

        Args:
            search_function: Function(query: str, top_k: int) -> List[Dict]
            verify_function: Optional Function(claim: str, context: List[str]) -> (bool, float)
        """
        self.search_function = search_function
        self.verify_function = verify_function

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Use custom search function."""
        return self.search_function(query, top_k)

    def verify_claim(self, claim: str, context: Optional[List[str]] = None) -> Tuple[bool, float]:
        """Use custom verify function or default implementation."""
        if self.verify_function:
            return self.verify_function(claim, context)
        
        # Default: search and check similarity
        results = self.search(claim, top_k=3)
        if not results:
            return True, 0.5

        avg_score = sum(r.get('score', 0.5) for r in results) / len(results)
        is_verified = avg_score > 0.7
        confidence = avg_score
        
        return is_verified, confidence
