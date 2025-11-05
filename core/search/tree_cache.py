import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib

from sentence_transformers import SentenceTransformer
from .lats import LATS, LATSNode


WORKER_THRESHOLDS = {
    "math": 0.90,
    "code": 0.87,
    "logic": 0.88,
    "factual": 0.80,
    "creative": 0.75,
    "analysis": 0.82
}


@dataclass
class CachedTree:
    query: str
    query_embedding: np.ndarray
    tree_id: str
    worker_specialty: str
    created_at: float
    success: bool
    confidence: float
    tree_path: str
    
    def to_dict(self) -> Dict[str, Any]:
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
        data['query_embedding'] = np.array(data['query_embedding'])
        return CachedTree(**data)


class TreeCache:
    def __init__(self, cache_dir: str = ".kaelum/tree_cache", similarity_threshold: float = 0.85, embedding_model: str = "all-MiniLM-L6-v2"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.trees_dir = self.cache_dir / "trees"
        self.trees_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.cache_dir / "metadata.json"
        self.similarity_threshold = similarity_threshold
        
        self.encoder = SentenceTransformer(embedding_model)
        self.cached_trees: List[CachedTree] = self._load_metadata()
        
    def _load_metadata(self) -> List[CachedTree]:
        if not self.metadata_file.exists():
            return []
        
        with open(self.metadata_file, 'r') as f:
            data = json.load(f)
        return [CachedTree.from_dict(item) for item in data]
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump([tree.to_dict() for tree in self.cached_trees], f, indent=2)
    
    def _compute_embedding(self, query: str) -> np.ndarray:
        return self.encoder.encode(query, show_progress_bar=False)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _generate_tree_id(self, query: str, worker_specialty: str) -> str:
        content = f"{query}_{worker_specialty}_{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def store(self, query_embedding: np.ndarray, cached_data: Dict[str, Any]) -> str:
        tree_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        tree_path = str(self.trees_dir / f"{tree_id}.json")
        
        with open(tree_path, 'w') as f:
            json.dump(cached_data, f, indent=2, default=str)
        
        cached_tree = CachedTree(
            query=cached_data.get("result", {}).get("query", ""),
            query_embedding=query_embedding,
            tree_id=tree_id,
            worker_specialty=cached_data.get("worker", ""),
            created_at=time.time(),
            success=cached_data.get("quality") == "high",
            confidence=cached_data.get("confidence", 0.0),
            tree_path=tree_path
        )
        
        self.cached_trees.append(cached_tree)
        self._save_metadata()
        
        return tree_id
    
    def get(self, query_embedding: np.ndarray, similarity_threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        if not self.cached_trees:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for cached_tree in self.cached_trees:
            similarity = self._cosine_similarity(query_embedding, cached_tree.query_embedding)
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = cached_tree
        
        if best_match is None or not best_match.success:
            return None
        
        with open(best_match.tree_path, 'r') as f:
            cached_data = json.load(f)
        
        return cached_data
    
    def get_stats(self) -> Dict[str, Any]:
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
        if worker_specialty:
            self.cached_trees = [t for t in self.cached_trees 
                                if t.worker_specialty != worker_specialty]
        else:
            self.cached_trees = []
        
        self._save_metadata()
