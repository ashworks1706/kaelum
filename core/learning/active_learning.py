from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import random
import torch
from sentence_transformers import SentenceTransformer, util
from core.shared_encoder import get_shared_encoder
from core.paths import DEFAULT_ACTIVE_LEARNING_DIR
import numpy as np


class QuerySelector:
    """Intelligent query selection for fine-tuning data collection.
    
    Strategies:
    1. Uncertainty sampling - Select queries where model is least confident
    2. Diversity sampling - Select queries that are semantically diverse
    3. Error-based sampling - Prioritize queries that failed verification
    4. Complexity sampling - Select queries with high reasoning complexity
    """
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', storage_dir: str = DEFAULT_ACTIVE_LEARNING_DIR):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.query_pool_file = self.storage_dir / "query_pool.jsonl"
        self.selected_file = self.storage_dir / "selected_queries.jsonl"
        self.embedding_model = SentenceTransformer(embedding_model)
        
        self.query_pool = []
        self.selected_queries = []
        self._load_pools()
    
    def _load_pools(self):
        if self.query_pool_file.exists():
            with open(self.query_pool_file, 'r') as f:
                for line in f:
                    try:
                        self.query_pool.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if self.selected_file.exists():
            with open(self.selected_file, 'r') as f:
                for line in f:
                    try:
                        self.selected_queries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    def add_query(self, query: str, result: Dict[str, Any]):
        """Add query and its execution result to the pool."""
        query_record = {
            "query": query,
            "worker": result.get("worker", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "verification_passed": result.get("verification_passed", False),
            "reasoning_steps": result.get("reasoning_trace", []),
            "answer": result.get("answer", ""),
            "timestamp": datetime.now().isoformat(),
            "complexity_score": self._compute_complexity(query, result),
            "selected": False
        }
        
        self.query_pool.append(query_record)
        
        with open(self.query_pool_file, 'a') as f:
            f.write(json.dumps(query_record) + '\n')
    
    def _compute_complexity(self, query: str, result: Dict[str, Any]) -> float:
        """Compute query complexity score based on multiple factors."""
        score = 0.0
        
        query_length = len(query.split())
        score += min(query_length / 50.0, 1.0) * 0.2
        
        num_steps = len(result.get("reasoning_trace", []))
        score += min(num_steps / 10.0, 1.0) * 0.3
        
        num_simulations = result.get("metrics", {}).get("num_simulations", 0)
        score += min(num_simulations / 20.0, 1.0) * 0.2
        
        tree_depth = result.get("metrics", {}).get("tree_depth", 0)
        score += min(tree_depth / 8.0, 1.0) * 0.2
        
        if not result.get("verification_passed", True):
            score += 0.1
        
        return min(score, 1.0)
    
    def select_by_uncertainty(self, n: int = 10) -> List[Dict[str, Any]]:
        """Select queries where model had lowest confidence."""
        unselected = [q for q in self.query_pool if not q.get("selected", False)]
        sorted_queries = sorted(unselected, key=lambda q: q["confidence"])
        return sorted_queries[:n]
    
    def select_by_diversity(self, n: int = 10) -> List[Dict[str, Any]]:
        unselected = [q for q in self.query_pool if not q.get("selected", False)]
        
        if len(unselected) <= n:
            return unselected
        
        queries_text = [q["query"] for q in unselected]
        embeddings = self.embedding_model.encode(queries_text, convert_to_tensor=True)
        
        selected_indices = []
        selected_indices.append(random.randint(0, len(unselected) - 1))
        
        selected_embeddings = embeddings[[selected_indices[0]]]
        
        while len(selected_indices) < n:
            unselected_mask = torch.ones(len(unselected), dtype=torch.bool, device=embeddings.device)
            unselected_mask[selected_indices] = False
            
            if not unselected_mask.any():
                break
            
            unselected_embs = embeddings[unselected_mask]
            similarities = util.pytorch_cos_sim(unselected_embs, selected_embeddings)
            
            max_similarities_per_candidate, _ = similarities.max(dim=1)
            
            min_max_similarity, best_relative_idx = max_similarities_per_candidate.min(dim=0)
            
            unselected_indices = torch.where(unselected_mask)[0]
            best_candidate = unselected_indices[best_relative_idx].item()
            
            selected_indices.append(best_candidate)
            selected_embeddings = embeddings[selected_indices]
        
        return [unselected[i] for i in selected_indices]
    
    def select_by_error(self, n: int = 10) -> List[Dict[str, Any]]:
        """Select queries that failed verification."""
        unselected = [q for q in self.query_pool if not q.get("selected", False)]
        failed = [q for q in unselected if not q.get("verification_passed", True)]
        
        sorted_by_complexity = sorted(failed, key=lambda q: q["complexity_score"], reverse=True)
        return sorted_by_complexity[:n]
    
    def select_by_complexity(self, n: int = 10) -> List[Dict[str, Any]]:
        """Select queries with highest reasoning complexity."""
        unselected = [q for q in self.query_pool if not q.get("selected", False)]
        sorted_queries = sorted(unselected, key=lambda q: q["complexity_score"], reverse=True)
        return sorted_queries[:n]
    
    def select_mixed(self, n: int = 20) -> List[Dict[str, Any]]:
        """Select queries using mixed strategy for balanced dataset."""
        uncertainty_queries = self.select_by_uncertainty(n // 4)
        diversity_queries = self.select_by_diversity(n // 4)
        error_queries = self.select_by_error(n // 4)
        complexity_queries = self.select_by_complexity(n // 4)
        
        all_selected = uncertainty_queries + diversity_queries + error_queries + complexity_queries
        
        unique_queries = []
        seen_ids = set()
        for q in all_selected:
            query_id = q["query"] + q["timestamp"]
            if query_id not in seen_ids:
                unique_queries.append(q)
                seen_ids.add(query_id)
        
        return unique_queries[:n]
    
    def mark_selected(self, queries: List[Dict[str, Any]]):
        """Mark queries as selected and save to selected file."""
        for query in queries:
            query["selected"] = True
            query["selected_at"] = datetime.now().isoformat()
            
            with open(self.selected_file, 'a') as f:
                f.write(json.dumps(query) + '\n')
        
        self._update_pool()
    
    def _update_pool(self):
        """Update the query pool file with selection status."""
        with open(self.query_pool_file, 'w') as f:
            for query in self.query_pool:
                f.write(json.dumps(query) + '\n')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the query pool."""
        total = len(self.query_pool)
        selected = len([q for q in self.query_pool if q.get("selected", False)])
        unselected = total - selected
        
        if total == 0:
            return {
                "total_queries": 0,
                "selected_queries": 0,
                "unselected_queries": 0,
                "avg_confidence": 0.0,
                "avg_complexity": 0.0,
                "verification_rate": 0.0
            }
        
        avg_confidence = sum(q["confidence"] for q in self.query_pool) / total
        avg_complexity = sum(q["complexity_score"] for q in self.query_pool) / total
        passed = len([q for q in self.query_pool if q.get("verification_passed", False)])
        
        return {
            "total_queries": total,
            "selected_queries": selected,
            "unselected_queries": unselected,
            "avg_confidence": avg_confidence,
            "avg_complexity": avg_complexity,
            "verification_rate": passed / total if total > 0 else 0.0,
            "by_worker": self._count_by_worker()
        }
    
    def _count_by_worker(self) -> Dict[str, int]:
        """Count queries by worker type."""
        counts = {}
        for q in self.query_pool:
            worker = q.get("worker", "unknown")
            counts[worker] = counts.get(worker, 0) + 1
        return counts


class ActiveLearningEngine:
    """Active learning engine for continuous model improvement."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", query_selector: Optional[QuerySelector] = None):
        self.query_selector = query_selector or QuerySelector(embedding_model=embedding_model)
        self.storage_dir = Path(DEFAULT_ACTIVE_LEARNING_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_file = self.storage_dir / "training_data.jsonl"
    
    def collect_query(self, query: str, result: Dict[str, Any]):
        """Collect query execution for potential fine-tuning."""
        self.query_selector.add_query(query, result)
    
    def generate_training_batch(
        self,
        strategy: str = "mixed",
        batch_size: int = 20
    ) -> List[Dict[str, Any]]:
        """Generate a batch of training data using specified strategy.
        
        Args:
            strategy: Selection strategy (uncertainty, diversity, error, complexity, mixed)
            batch_size: Number of queries to select
        
        Returns:
            List of selected queries formatted for training
        """
        if strategy == "uncertainty":
            selected = self.query_selector.select_by_uncertainty(batch_size)
        elif strategy == "diversity":
            selected = self.query_selector.select_by_diversity(batch_size)
        elif strategy == "error":
            selected = self.query_selector.select_by_error(batch_size)
        elif strategy == "complexity":
            selected = self.query_selector.select_by_complexity(batch_size)
        else:
            selected = self.query_selector.select_mixed(batch_size)
        
        self.query_selector.mark_selected(selected)
        
        training_examples = []
        for query_record in selected:
            example = self._format_for_training(query_record)
            training_examples.append(example)
            
            with open(self.training_data_file, 'a') as f:
                f.write(json.dumps(example) + '\n')
        
        return training_examples
    
    def _format_for_training(self, query_record: Dict[str, Any]) -> Dict[str, Any]:
        """Format query record for fine-tuning."""
        reasoning_text = "\n".join([
            f"{i+1}. {step}" 
            for i, step in enumerate(query_record["reasoning_steps"])
        ])
        
        return {
            "query": query_record["query"],
            "reasoning": reasoning_text,
            "answer": query_record["answer"],
            "worker_type": query_record["worker"],
            "complexity": query_record["complexity_score"],
            "verified": query_record["verification_passed"]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get active learning statistics."""
        return self.query_selector.get_statistics()
    
    def export_training_dataset(self, output_path: str):
        """Export all selected queries as a training dataset."""
        training_data = []
        
        if self.training_data_file.exists():
            with open(self.training_data_file, 'r') as f:
                training_data = [json.loads(line) for line in f]
        
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        return len(training_data)
