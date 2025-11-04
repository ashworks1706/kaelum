"""Intelligent Router using embeddings and learning - NO keyword matching.

This router uses semantic understanding via embeddings to classify queries.
It learns from outcomes to improve over time.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from kaelum.core.router import QueryType, ReasoningStrategy, RoutingDecision


@dataclass
class TrainingExample:
    """A training example for the router."""
    query: str
    query_type: QueryType
    strategy: ReasoningStrategy
    embedding: Optional[np.ndarray] = None
    performance_score: float = 0.0  # 0-1, how well this routing worked


class IntelligentRouter:
    """Router that uses embeddings and learns from outcomes.
    
    NO keyword matching. Uses semantic similarity to classify queries.
    Learns from every routing decision to improve over time.
    """
    
    def __init__(self, data_dir: str = ".kaelum/intelligent_routing"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load lightweight but effective embedding model (80MB)
        print("Loading embedding model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Embedding model loaded")
        
        # Load training examples
        self.training_file = self.data_dir / "training_examples.jsonl"
        self.training_examples = self._load_training_examples()
        
        # Compute embeddings for all training examples
        if self.training_examples:
            self._compute_embeddings()
        else:
            # Bootstrap with initial examples
            self._bootstrap_initial_examples()
        
        print(f"✓ Router initialized with {len(self.training_examples)} training examples")
    
    def route(self, query: str) -> RoutingDecision:
        """Route query using semantic similarity to training examples.
        
        Args:
            query: Input query to route
            
        Returns:
            RoutingDecision with strategy and configuration
        """
        # Encode query
        query_embedding = self.encoder.encode([query])[0]
        
        # Find K nearest neighbors in training data
        k = 10
        neighbors = self._find_nearest_neighbors(query_embedding, k)
        
        # Weighted vote based on similarity and performance
        query_type, type_confidence = self._weighted_vote_query_type(neighbors)
        strategy = self._select_strategy(query_type, neighbors)
        
        # Build configuration
        config = self._build_config(query_type, strategy, type_confidence)
        
        print(f"\n🧠 INTELLIGENT ROUTING")
        print(f"   Query: {query[:80]}...")
        print(f"   Type: {query_type.value} (confidence: {type_confidence:.2f})")
        print(f"   Strategy: {strategy.value}")
        print(f"   Based on {len(neighbors)} similar examples")
        
        return RoutingDecision(
            query_type=query_type,
            strategy=strategy,
            max_reflection_iterations=config['max_reflection'],
            use_symbolic_verification=config['use_symbolic'],
            use_factual_verification=config['use_factual'],
            confidence_threshold=config['confidence_threshold'],
            reasoning=f"Semantic similarity to {len(neighbors)} examples, confidence {type_confidence:.2f}",
            complexity_score=self._estimate_complexity(query)
        )
    
    def learn_from_outcome(self, query: str, decision: RoutingDecision, 
                          accuracy: float, latency_ms: float):
        """Learn from routing outcome to improve future decisions.
        
        Args:
            query: The query that was routed
            decision: The routing decision made
            accuracy: How accurate the result was (0-1)
            latency_ms: How long it took
        """
        # Only learn from good outcomes
        if accuracy < 0.6:
            return
        
        # Calculate performance score (balance accuracy and speed)
        performance_score = accuracy * 0.8 + (1 - min(latency_ms / 5000, 1)) * 0.2
        
        # Create training example
        example = TrainingExample(
            query=query,
            query_type=decision.query_type,
            strategy=decision.strategy,
            performance_score=performance_score
        )
        
        # Compute embedding
        example.embedding = self.encoder.encode([query])[0]
        
        # Add to training data
        self.training_examples.append(example)
        
        # Save to disk
        self._save_example(example)
        
        # Recompute embeddings if we have enough new examples
        if len(self.training_examples) % 20 == 0:
            print(f"✓ Router learned from {len(self.training_examples)} examples")
    
    # ==================== INTERNAL METHODS ====================
    
    def _find_nearest_neighbors(self, query_embedding: np.ndarray, k: int) -> List[TrainingExample]:
        """Find K nearest training examples."""
        if not self.training_examples:
            return []
        
        # Get all embeddings
        training_embeddings = np.array([ex.embedding for ex in self.training_examples])
        
        # Compute similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            training_embeddings
        )[0]
        
        # Get top K
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.training_examples[i] for i in top_k_indices]
    
    def _weighted_vote_query_type(self, neighbors: List[TrainingExample]) -> Tuple[QueryType, float]:
        """Vote on query type weighted by similarity and performance."""
        if not neighbors:
            return QueryType.UNKNOWN, 0.5
        
        # Weight by performance score
        type_scores = defaultdict(float)
        for example in neighbors:
            type_scores[example.query_type] += example.performance_score
        
        # Normalize
        total_score = sum(type_scores.values())
        if total_score > 0:
            type_scores = {k: v/total_score for k, v in type_scores.items()}
        
        # Get best type and confidence
        best_type = max(type_scores.items(), key=lambda x: x[1])
        return best_type[0], best_type[1]
    
    def _select_strategy(self, query_type: QueryType, neighbors: List[TrainingExample]) -> ReasoningStrategy:
        """Select strategy based on query type and historical performance."""
        # Find successful strategies for this query type
        relevant_examples = [ex for ex in neighbors if ex.query_type == query_type]
        
        if not relevant_examples:
            # Default strategies per type
            defaults = {
                QueryType.MATH: ReasoningStrategy.SYMBOLIC_HEAVY,
                QueryType.LOGIC: ReasoningStrategy.DEEP,
                QueryType.CODE: ReasoningStrategy.BALANCED,
                QueryType.FACTUAL: ReasoningStrategy.FACTUAL_HEAVY,
                QueryType.CREATIVE: ReasoningStrategy.BALANCED,
                QueryType.ANALYSIS: ReasoningStrategy.DEEP,
                QueryType.UNKNOWN: ReasoningStrategy.BALANCED
            }
            return defaults.get(query_type, ReasoningStrategy.BALANCED)
        
        # Weighted vote on strategy
        strategy_scores = defaultdict(float)
        for example in relevant_examples:
            strategy_scores[example.strategy] += example.performance_score
        
        return max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    def _build_config(self, query_type: QueryType, strategy: ReasoningStrategy, 
                     confidence: float) -> Dict:
        """Build configuration for this query type and strategy."""
        config = {
            'max_reflection': 2,
            'use_symbolic': False,
            'use_factual': False,
            'confidence_threshold': 0.7
        }
        
        # Strategy-specific config
        if strategy == ReasoningStrategy.DEEP:
            config['max_reflection'] = 3
            config['confidence_threshold'] = 0.8
        elif strategy == ReasoningStrategy.FAST:
            config['max_reflection'] = 1
            config['confidence_threshold'] = 0.6
        elif strategy == ReasoningStrategy.SYMBOLIC_HEAVY:
            config['use_symbolic'] = True
            config['max_reflection'] = 2
        elif strategy == ReasoningStrategy.FACTUAL_HEAVY:
            config['use_factual'] = True
            config['max_reflection'] = 2
        
        # Query type specific adjustments
        if query_type == QueryType.MATH:
            config['use_symbolic'] = True
        elif query_type == QueryType.FACTUAL:
            config['use_factual'] = True
        
        return config
    
    def _estimate_complexity(self, query: str) -> float:
        """Estimate query complexity (0-1)."""
        # Simple heuristic: length and structural complexity
        words = len(query.split())
        sentences = query.count('.') + query.count('?') + 1
        
        # Normalize to 0-1
        complexity = min((words / 50) * 0.6 + (sentences / 5) * 0.4, 1.0)
        return complexity
    
    def _compute_embeddings(self):
        """Compute embeddings for all training examples that don't have them."""
        queries_to_encode = []
        indices_to_update = []
        
        for i, example in enumerate(self.training_examples):
            if example.embedding is None:
                queries_to_encode.append(example.query)
                indices_to_update.append(i)
        
        if queries_to_encode:
            embeddings = self.encoder.encode(queries_to_encode)
            for idx, embedding in zip(indices_to_update, embeddings):
                self.training_examples[idx].embedding = embedding
    
    def _bootstrap_initial_examples(self):
        """Bootstrap with initial training examples."""
        initial_examples = [
            # Math examples
            ("Calculate 15 * 23", QueryType.MATH, ReasoningStrategy.SYMBOLIC_HEAVY),
            ("What is 2 + 2?", QueryType.MATH, ReasoningStrategy.FAST),
            ("Solve for x: 3x + 7 = 22", QueryType.MATH, ReasoningStrategy.SYMBOLIC_HEAVY),
            ("Find the derivative of x^2 + 3x", QueryType.MATH, ReasoningStrategy.SYMBOLIC_HEAVY),
            
            # Logic examples
            ("If all A are B, and all B are C, are all A also C?", QueryType.LOGIC, ReasoningStrategy.DEEP),
            ("Is this argument valid?", QueryType.LOGIC, ReasoningStrategy.DEEP),
            
            # Code examples
            ("Write a Python function to reverse a string", QueryType.CODE, ReasoningStrategy.BALANCED),
            ("Debug this code", QueryType.CODE, ReasoningStrategy.BALANCED),
            
            # Factual examples
            ("What is the capital of France?", QueryType.FACTUAL, ReasoningStrategy.FACTUAL_HEAVY),
            ("When did World War 2 end?", QueryType.FACTUAL, ReasoningStrategy.FACTUAL_HEAVY),
            
            # Creative examples
            ("Write a short story about a robot", QueryType.CREATIVE, ReasoningStrategy.BALANCED),
            ("Brainstorm ideas for a new app", QueryType.CREATIVE, ReasoningStrategy.BALANCED),
        ]
        
        for query, qtype, strategy in initial_examples:
            example = TrainingExample(
                query=query,
                query_type=qtype,
                strategy=strategy,
                performance_score=0.8  # Default performance
            )
            example.embedding = self.encoder.encode([query])[0]
            self.training_examples.append(example)
        
        print(f"✓ Bootstrapped with {len(initial_examples)} initial examples")
    
    def _load_training_examples(self) -> List[TrainingExample]:
        """Load training examples from disk."""
        if not self.training_file.exists():
            return []
        
        examples = []
        with open(self.training_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                example = TrainingExample(
                    query=data['query'],
                    query_type=QueryType[data['query_type']],
                    strategy=ReasoningStrategy[data['strategy']],
                    embedding=np.array(data['embedding']) if 'embedding' in data else None,
                    performance_score=data.get('performance_score', 0.8)
                )
                examples.append(example)
        
        return examples
    
    def _save_example(self, example: TrainingExample):
        """Save training example to disk."""
        data = {
            'query': example.query,
            'query_type': example.query_type.value,
            'strategy': example.strategy.value,
            'embedding': example.embedding.tolist() if example.embedding is not None else None,
            'performance_score': example.performance_score
        }
        
        with open(self.training_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
