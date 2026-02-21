import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib

from sentence_transformers import SentenceTransformer
from .lats import LATS, LATSNode
from core.cache_validator import CacheValidator
from core.shared_encoder import get_shared_encoder
from core.paths import DEFAULT_TREE_CACHE_DIR

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

WORKER_THRESHOLDS = {
# Worker-specific similarity thresholds for cache hits
# Higher thresholds for math/code (require exact matches) 
# Lower thresholds for creative/factual (more semantic flexibility)
# These values tuned empirically based on false positive rates
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

        data.pop('query_fingerprint', None)
        return CachedTree(**data)

class TreeCache:
    def __init__(self, cache_dir: str = DEFAULT_TREE_CACHE_DIR, similarity_threshold: float = 0.85, 
                 embedding_model: str = "all-MiniLM-L6-v2", llm_client=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.trees_dir = self.cache_dir / "trees"
        self.trees_dir.mkdir(exist_ok=True)
        
        self.metadata_path = self.cache_dir / "metadata.json"
        self.similarity_threshold = similarity_threshold
        
        self.max_cache_size = 1000
        self.use_faiss = False
        self.faiss_index = None
        
        self.encoder = get_shared_encoder(embedding_model, device='cpu')
        self.validator = CacheValidator(llm_client=llm_client)
        self.cached_trees: List[CachedTree] = self._load_metadata()
        self.access_times: Dict[str, float] = {}
        
    def _load_metadata(self) -> List[CachedTree]:
        if not self.metadata_path.exists():
            return []
        
        cached_trees = []
        with open(self.metadata_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                cached_trees.append(CachedTree.from_dict(item))
        return cached_trees
    
    def _save_metadata(self):
        with open(self.metadata_path, 'w') as f:
            for tree in self.cached_trees:
                f.write(json.dumps(tree.to_dict()) + '\n')
    
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
        import logging
        logger = logging.getLogger("kaelum.cache")
        
        tree_id = f"tree_{int(time.time() * 1000000)}"
        tree_path = str(self.trees_dir / f"{tree_id}.json")
        
        with open(tree_path, 'w') as f:
            json.dump(cached_data, f, indent=2)
        
        result = cached_data.get("result", {})
        worker = result.get("worker", "unknown")
        confidence = result.get("confidence", 0.0)
        success = result.get("verification_passed", False)
        
        cached_tree = CachedTree(
            tree_id=tree_id,
            tree_path=tree_path,
            query=result.get("query", ""),
            query_embedding=query_embedding,
            worker_specialty=worker,
            confidence=confidence,
            success=success,
            created_at=time.time()
        )
        
        self.cached_trees.append(cached_tree)
        self.access_times[tree_id] = time.time()
        
        if len(self.cached_trees) > self.max_cache_size:
            self._evict_entries()
        
        if self.use_faiss or (FAISS_AVAILABLE and len(self.cached_trees) > 50):
            self.use_faiss = True
            self._build_faiss_index()
        
        self._save_metadata()
        
        logger.info(f"CACHE: Stored tree {tree_id} (worker={worker}, confidence={confidence:.3f}, success={success})")
        
        return tree_id
    
    def _evict_entries(self):
        import logging
        logger = logging.getLogger("kaelum.cache")
        
        target_size = int(self.max_cache_size * 0.8)
        num_to_evict = len(self.cached_trees) - target_size
        
        if num_to_evict <= 0:
            return
        
        scored_trees = []
        for tree in self.cached_trees:
            quality_score = tree.confidence if tree.success else tree.confidence * 0.3
            
            recency_score = self.access_times.get(tree.tree_id, tree.created_at)
            age = time.time() - recency_score
            recency_score = 1.0 / (1.0 + age / 86400.0)
            
            combined_score = quality_score * 0.7 + recency_score * 0.3
            
            scored_trees.append((tree, combined_score))
        
        scored_trees.sort(key=lambda x: x[1])
        
        to_evict = scored_trees[:num_to_evict]
        
        for tree, score in to_evict:
            tree_path = Path(tree.tree_path)
            if tree_path.exists():
                tree_path.unlink()
            
            if tree.tree_id in self.access_times:
                del self.access_times[tree.tree_id]
        
        evicted_ids = {tree.tree_id for tree, _ in to_evict}
        self.cached_trees = [t for t in self.cached_trees if t.tree_id not in evicted_ids]
        
        if self.use_faiss:
            self._build_faiss_index()
        
        logger.info(f"CACHE EVICTION: Removed {num_to_evict} low-quality/old entries ({len(self.cached_trees)}/{self.max_cache_size} remain)")
    
    def _build_faiss_index(self):
        if not self.cached_trees or not FAISS_AVAILABLE:
            self.faiss_index = None
            return
        
        import logging
        logger = logging.getLogger("kaelum.cache")
        
        embeddings = np.array([tree.query_embedding for tree in self.cached_trees]).astype('float32')
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        
        logger.info(f"CACHE: Built FAISS index with {len(self.cached_trees)} entries")
    
    def _check_symbolic_equivalence(self, query1: str, query2: str, worker: str) -> bool:
        if worker != "math":
            return False
        
        from sympy import sympify, simplify
        import re
        
        def extract_expression(text):
            expr_patterns = [
                r'derivative\s+of\s+([^?\.]+)',
                r'solve\s+([^?\.]+)',
                r'integrate\s+([^?\.]+)',
                r'simplify\s+([^?\.]+)',
                r'([x\+\-\*\/\^\d\(\)]+)\s*[=\?]?'
            ]
            
            for pattern in expr_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    return match.group(1).strip()
            return None
        
        expr1 = extract_expression(query1)
        expr2 = extract_expression(query2)
        
        if not expr1 or not expr2:
            return False
        
        sym_expr1 = sympify(expr1)
        sym_expr2 = sympify(expr2)
        diff = simplify(sym_expr1 - sym_expr2)
        return diff == 0
    
    def get(self, query: str, query_embedding: np.ndarray, similarity_threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        import logging
        logger = logging.getLogger("kaelum.cache")
        
        if not self.cached_trees:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        if self.use_faiss and self.faiss_index is not None:
            query_emb_normalized = query_embedding.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_emb_normalized)
            
            k = min(10, len(self.cached_trees))
            similarities, indices = self.faiss_index.search(query_emb_normalized, k)
            
            for i, idx in enumerate(indices[0]):
                if idx == -1:
                    break
                
                cached_tree = self.cached_trees[idx]
                if not cached_tree.success:
                    continue
                
                similarity = float(similarities[0][i])
                
                if similarity < similarity_threshold:
                    continue
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached_tree
        else:
            for cached_tree in self.cached_trees:
                if not cached_tree.success:
                    continue
                
                similarity = self._cosine_similarity(query_embedding, cached_tree.query_embedding)
                
                if similarity < similarity_threshold:
                    continue
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached_tree
        
        if best_match is None:
            logger.debug(f"CACHE: No similarity match found (threshold={similarity_threshold})")
            return None
        
        self.access_times[best_match.tree_id] = time.time()
        
        logger.info(f"\nCACHE: Similarity match found!")
        logger.info(f"  Similarity: {best_similarity:.3f}")
        logger.info(f"  Cached query: {best_match.query[:100]}...")
        logger.info(f"  Worker: {best_match.worker_specialty}")
        
        if best_similarity > 0.95:
            logger.info(f"CACHE: ✓ HIT (very high similarity, skip validation)")
            with open(best_match.tree_path, 'r') as f:
                return json.load(f)
        
        if best_similarity < 0.90:
            symbolic_equivalent = self._check_symbolic_equivalence(
                query, best_match.query, best_match.worker_specialty
            )
            
            if symbolic_equivalent:
                logger.info(f"CACHE: ✓ HIT (symbolically equivalent)")
                with open(best_match.tree_path, 'r') as f:
                    return json.load(f)
        
        with open(best_match.tree_path, 'r') as f:
            cached_data = json.load(f)
        
        cached_answer = cached_data.get('result', {}).get('answer', '')
        validation = self.validator.validate_cache_match(
            query, 
            best_match.query, 
            cached_answer
        )
        
        if not validation.get('valid', False):
            logger.info(f"CACHE: ✗ MISS (LLM validation rejected)")
            return None
        
        logger.info(f"CACHE: ✓ HIT (LLM validated)")
        return cached_data
    

