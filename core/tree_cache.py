"""Tree caching system for LATS reasoning trees.

Caches reasoning trees based on query similarity to enable:
1. Fast retrieval of similar past reasoning
2. Transfer learning across similar queries
3. Reduced computation for repeated query types
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

from .lats import LATS, LATSNode


@dataclass
class CachedTree:
    """Cached reasoning tree metadata."""
    query: str
    query_embedding: np.ndarray
    tree_id: str
    worker_specialty: str
    created_at: float
    success: bool
    confidence: float
    tree_path: str  # Path to saved LATS tree JSON
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'query': self.query,
            'query_embedding': self.query_embedding.tolist(),
            'tree_id': self.tree_id,
            'worker_specialty': self.worker_specialty,
            'created_at': self.created_at,
            'success': self.success,
            'confidence': self.confidence,
            'tree_path': self.tree_path
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'CachedTree':
        """Create from dictionary."""
        data['query_embedding'] = np.array(data['query_embedding'])
        return CachedTree(**data)


class TreeCache:
    """Cache for LATS reasoning trees with similarity-based retrieval.
    
    Architecture:
    - Stores trees on disk as JSON
    - Maintains metadata index with embeddings
    - Retrieves trees by cosine similarity
    - Supports filtering by worker specialty
    """
    
    def __init__(self, cache_dir: str = ".kaelum/tree_cache", similarity_threshold: float = 0.85):
        """Initialize tree cache.
        
        Args:
            cache_dir: Directory to store cached trees
            similarity_threshold: Minimum cosine similarity to consider a hit (0-1)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.trees_dir = self.cache_dir / "trees"
        self.trees_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.cache_dir / "metadata.json"
        self.similarity_threshold = similarity_threshold
        
        # Initialize encoder for similarity search
        self.encoder = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass
        
        # Load existing cache metadata
        self.cached_trees: List[CachedTree] = self._load_metadata()
        
    def _load_metadata(self) -> List[CachedTree]:
        """Load cached tree metadata from disk."""
        if not self.metadata_file.exists():
            return []
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            return [CachedTree.from_dict(item) for item in data]
        except Exception:
            return []
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump([tree.to_dict() for tree in self.cached_trees], f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache metadata: {e}")
    
    def _compute_embedding(self, query: str) -> np.ndarray:
        """Compute query embedding for similarity search."""
        if self.encoder is not None:
            return self.encoder.encode(query, show_progress_bar=False)
        else:
            # Fallback: zero embedding
            return np.zeros(384, dtype=np.float32)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _generate_tree_id(self, query: str, worker_specialty: str) -> str:
        """Generate unique ID for a tree."""
        content = f"{query}_{worker_specialty}_{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def store(self, query: str, tree: LATS, worker_specialty: str, 
              success: bool, confidence: float) -> str:
        """Store a reasoning tree in the cache.
        
        Args:
            query: Original query
            tree: LATS tree to cache
            worker_specialty: Worker specialty that created this tree
            success: Whether the reasoning was successful
            confidence: Confidence score (0-1)
            
        Returns:
            Tree ID
        """
        # Generate tree ID and paths
        tree_id = self._generate_tree_id(query, worker_specialty)
        tree_path = str(self.trees_dir / f"{tree_id}.json")
        
        # Save tree to disk
        tree.save(tree_path)
        
        # Compute embedding
        query_embedding = self._compute_embedding(query)
        
        # Create metadata entry
        cached_tree = CachedTree(
            query=query,
            query_embedding=query_embedding,
            tree_id=tree_id,
            worker_specialty=worker_specialty,
            created_at=time.time(),
            success=success,
            confidence=confidence,
            tree_path=tree_path
        )
        
        # Add to cache
        self.cached_trees.append(cached_tree)
        self._save_metadata()
        
        return tree_id
    
    def retrieve(self, query: str, worker_specialty: Optional[str] = None,
                 require_success: bool = True) -> Optional[Tuple[LATS, CachedTree, float]]:
        """Retrieve a similar reasoning tree from cache.
        
        Args:
            query: Query to search for
            worker_specialty: Optional filter by worker specialty
            require_success: Only return successful trees
            
        Returns:
            Tuple of (LATS tree, CachedTree metadata, similarity_score) or None
        """
        if not self.cached_trees:
            return None
        
        # Compute query embedding
        query_embedding = self._compute_embedding(query)
        
        # Find most similar cached tree
        best_match = None
        best_similarity = 0.0
        
        for cached_tree in self.cached_trees:
            # Apply filters
            if worker_specialty and cached_tree.worker_specialty != worker_specialty:
                continue
            if require_success and not cached_tree.success:
                continue
            
            # Compute similarity
            similarity = self._cosine_similarity(query_embedding, cached_tree.query_embedding)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = cached_tree
        
        if best_match is None:
            return None
        
        # Load tree from disk
        try:
            tree = LATS.load(best_match.tree_path)
            return tree, best_match, best_similarity
        except Exception:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cached_trees:
            return {
                'total_trees': 0,
                'by_specialty': {},
                'avg_confidence': 0.0,
                'success_rate': 0.0
            }
        
        by_specialty = {}
        for tree in self.cached_trees:
            spec = tree.worker_specialty
            if spec not in by_specialty:
                by_specialty[spec] = {'count': 0, 'success': 0}
            by_specialty[spec]['count'] += 1
            if tree.success:
                by_specialty[spec]['success'] += 1
        
        avg_confidence = sum(t.confidence for t in self.cached_trees) / len(self.cached_trees)
        success_rate = sum(1 for t in self.cached_trees if t.success) / len(self.cached_trees)
        
        return {
            'total_trees': len(self.cached_trees),
            'by_specialty': by_specialty,
            'avg_confidence': avg_confidence,
            'success_rate': success_rate
        }
    
    def clear(self, worker_specialty: Optional[str] = None):
        """Clear cache, optionally filtered by specialty."""
        if worker_specialty:
            self.cached_trees = [t for t in self.cached_trees 
                                if t.worker_specialty != worker_specialty]
        else:
            self.cached_trees = []
        
        self._save_metadata()
