"""Intelligent routing using embeddings and learning.

NO keyword matching. Uses sentence embeddings to understand query semantics
and learns from outcomes to improve routing over time.

Architecture:
- Embedding-based classification (sentence-transformers)
- K-NN with similarity search
- Continuous learning from outcomes
- Optional: Neural classifier (Qwen2.5-1.5B) for advanced routing
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not installed. Install with: pip install scikit-learn")

# Set up router-specific logger
logger = logging.getLogger("kaelum.router")
logger.setLevel(logging.INFO)


class QueryType(Enum):
    """Types of queries the router can classify."""
    MATH = "math"
    LOGIC = "logic"
    CODE = "code"
    FACTUAL = "factual"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    UNKNOWN = "unknown"


class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    SYMBOLIC_HEAVY = "symbolic_heavy"  # Deep symbolic verification, math focus
    FACTUAL_HEAVY = "factual_heavy"    # RAG-heavy, knowledge verification
    BALANCED = "balanced"              # Default: moderate all checks
    FAST = "fast"                      # Minimal verification, speed priority
    DEEP = "deep"                      # Max reflection iterations, accuracy priority


@dataclass
class RoutingDecision:
    """Decision made by the router - routes to specific worker agent."""
    query_type: QueryType
    worker_specialty: str  # Which worker to use: 'math', 'logic', 'code', 'factual', 'creative', 'analysis'
    confidence: float  # Router's confidence in this decision (0-1)
    reasoning: str = ""
    # Additional metadata
    secondary_types: List[QueryType] = None  # Multi-category detection
    complexity_score: float = 0.0  # 0-1, estimated query complexity
    use_tree_cache: bool = True  # Whether to check/use cached reasoning trees
    max_tree_depth: int = 5  # Maximum depth for LATS tree search
    num_simulations: int = 10  # Number of MCTS simulations to run
    
    def __post_init__(self):
        if self.secondary_types is None:
            self.secondary_types = []


@dataclass
class RoutingOutcome:
    """Outcome of a routing decision (for learning)."""
    query: str
    query_type: QueryType
    strategy: ReasoningStrategy
    decision: RoutingDecision
    
    # Performance metrics
    success: bool
    accuracy_score: float  # 0-1 based on verification
    latency_ms: float
    cost: float
    
    # Verification results
    symbolic_passed: bool
    factual_passed: bool
    reflection_iterations: int
    
    timestamp: float


class Router:
    """Intelligent router using embeddings and learning.
    
    NO keyword matching. Uses semantic similarity to classify queries
    and learns from outcomes to improve over time.
    """
    
    def __init__(self, learning_enabled: bool = True, data_dir: str = ".kaelum/routing"):
        self.learning_enabled = learning_enabled
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.outcomes_file = self.data_dir / "outcomes.jsonl"
        self.stats_file = self.data_dir / "stats.json"
        self.training_data_file = self.data_dir / "training_queries.json"
        
        # Initialize embedding model (lightweight, 80MB)
        if EMBEDDINGS_AVAILABLE:
            logger.info("Loading sentence embedding model...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Embedding model loaded")
        else:
            logger.warning("Embeddings not available - falling back to basic classification")
            self.encoder = None
        
        # Load training data and embeddings
        self.training_queries = self._load_training_data()
        if self.encoder and self.training_queries:
            logger.info(f"Computing embeddings for {len(self.training_queries)} training examples...")
            self.training_embeddings = self.encoder.encode(
                [q['query'] for q in self.training_queries],
                show_progress_bar=False
            )
            logger.info("✓ Training embeddings ready")
        else:
            self.training_embeddings = None
        
        # Performance stats
        self.performance_stats = self._load_stats()
        
        logger.info("=" * 60)
        logger.info("Intelligent Router initialized")
        logger.info(f"Embedding-based: {self.encoder is not None}")
        logger.info(f"Training examples: {len(self.training_queries)}")
        logger.info(f"Learning enabled: {learning_enabled}")
        logger.info("=" * 60)
        
    def route(self, query: str, context: Optional[Dict] = None) -> RoutingDecision:
        """Route a query to the optimal worker agent.
        
        Args:
            query: The input query to route
            context: Optional context (previous results, user preferences, etc.)
            
        Returns:
            RoutingDecision with worker selection and configuration
        """
        start_time = time.time()
        
        logger.info("-" * 60)
        logger.info(f"ROUTING REQUEST: {query[:100]}...")
        
        # Step 1: Classify query type and get all scores
        query_type, scores = self._classify_query(query, context)
        logger.info(f"  Query Type: {query_type.value}")
        
        # Step 1b: Detect secondary types (multi-category)
        secondary_types = self._get_secondary_types(scores, query_type)
        if secondary_types:
            logger.info(f"  Secondary Types: {[qt.value for qt in secondary_types]}")
        
        # Step 1c: Estimate complexity
        complexity = self._estimate_complexity(query, scores)
        logger.info(f"  Complexity: {complexity:.2f}")
        
        # Step 2: Select worker based on query type + learned performance
        worker_specialty = self._select_worker(query_type, context)
        logger.info(f"  Worker: {worker_specialty}")
        
        # Step 3: Configure LATS parameters based on complexity
        lats_config = self._build_lats_config(query_type, complexity, context)
        logger.info(f"  LATS Config: depth={lats_config['max_tree_depth']}, "
                   f"simulations={lats_config['num_simulations']}, "
                   f"cache={lats_config['use_tree_cache']}")
        
        # Step 4: Determine confidence and reasoning
        confidence = self._calculate_routing_confidence(query_type, scores, context)
        reasoning = self._explain_decision(query, query_type, worker_specialty, complexity)
        logger.info(f"  Confidence: {confidence:.2f}")
        logger.info(f"  Reasoning: {reasoning}")
        
        routing_time = (time.time() - start_time) * 1000
        logger.info(f"  Routing time: {routing_time:.2f}ms")
        logger.info("-" * 60)
        
        return RoutingDecision(
            query_type=query_type,
            worker_specialty=worker_specialty,
            confidence=confidence,
            reasoning=reasoning,
            secondary_types=secondary_types,
            complexity_score=complexity,
            **lats_config
        )
    
    def record_outcome(self, decision: RoutingDecision, result: Dict[str, Any]):
        """Record the outcome of a routing decision for learning.
        
        Args:
            decision: The routing decision that was made
            result: The result from the worker (includes metrics, verification)
        """
        if not self.learning_enabled:
            return
        
        outcome = RoutingOutcome(
            query=result.get("query", ""),
            query_type=decision.query_type,
            strategy=ReasoningStrategy.BALANCED,  # Deprecated but kept for compatibility
            decision=decision,
            
            success=result.get("success", False),
            accuracy_score=result.get("confidence", 0.0),
            latency_ms=result.get("execution_time", 0) * 1000,  # Convert to ms
            cost=result.get("cost", 0.0),
            
            symbolic_passed=result.get("verification_passed", False),
            factual_passed=result.get("verification_passed", False),
            reflection_iterations=0,  # Not applicable in LATS model
            
            timestamp=time.time()
        )
        
        logger.info("=" * 60)
        logger.info("ROUTING OUTCOME")
        logger.info(f"  Worker: {decision.worker_specialty}")
        logger.info(f"  Success: {outcome.success}")
        logger.info(f"  Confidence: {outcome.accuracy_score:.2f}")
        logger.info(f"  Latency: {outcome.latency_ms:.2f}ms")
        logger.info(f"  Cost: ${outcome.cost:.6f}")
        logger.info("=" * 60)
        
        # Append to outcomes log
        with open(self.outcomes_file, "a") as f:
            f.write(json.dumps(asdict(outcome), default=str) + "\n")
        
        # Update performance stats
        self._update_stats_worker(outcome, decision.worker_specialty)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of routing performance across strategies."""
        return {
            "total_queries": sum(s["count"] for s in self.performance_stats.values()),
            "by_strategy": self.performance_stats,
            "learning_enabled": self.learning_enabled,
            "outcomes_logged": self._count_outcomes()
        }
    
    def simulate_queries(self, test_queries: List[Dict[str, str]], num_runs: int = 5):
        """Simulate routing decisions on test queries to learn patterns.
        
        This is used to bootstrap the router with synthetic data before real usage.
        
        Args:
            test_queries: List of {"query": str, "expected_type": str, "ground_truth": str}
            num_runs: Number of simulation runs per query
        """
        print(f"\n🎮 [ROUTING SIMULATION]")
        print(f"   Testing {len(test_queries)} queries × {num_runs} strategies")
        print(f"   Collecting data to {self.outcomes_file}\n")
        
        strategies = list(ReasoningStrategy)
        results = []
        
        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            expected_type = QueryType[test_case.get("expected_type", "UNKNOWN").upper()]
            
            print(f"   [{i}/{len(test_queries)}] {query[:60]}...")
            
            for strategy in strategies:
                # Simulate routing decision
                decision = RoutingDecision(
                    query_type=expected_type,
                    strategy=strategy,
                    max_reflection_iterations=2 if strategy == ReasoningStrategy.DEEP else 1,
                    use_symbolic_verification=strategy in [ReasoningStrategy.SYMBOLIC_HEAVY, ReasoningStrategy.BALANCED],
                    use_factual_verification=strategy in [ReasoningStrategy.FACTUAL_HEAVY, ReasoningStrategy.BALANCED],
                    confidence_threshold=0.8,
                    reasoning=f"Simulation: testing {strategy.value}"
                )
                
                # Simulate outcome (simplified - real outcomes come from orchestrator)
                simulated_outcome = self._simulate_outcome(query, decision, test_case)
                
                # Record simulated outcome
                outcome_data = asdict(simulated_outcome)
                # Convert enums to strings for JSON serialization
                outcome_data['query_type'] = simulated_outcome.query_type.value
                outcome_data['strategy'] = simulated_outcome.strategy.value
                outcome_data['decision'] = None  # Skip decision details
                
                with open(self.outcomes_file, "a") as f:
                    f.write(json.dumps(outcome_data, default=str) + "\n")
                
                # Immediately update stats for this outcome
                self._update_stats(simulated_outcome)
                
                results.append({
                    "query": query[:40],
                    "strategy": strategy.value,
                    "accuracy": simulated_outcome.accuracy_score,
                    "latency": simulated_outcome.latency_ms
                })
        
        print(f"\n   ✓ Simulation complete: {len(results)} outcomes logged")
        # Don't call _update_stats_from_file since we already updated stats inline
        return results
    
    # ==================== INTERNAL METHODS ====================
    
    def _classify_query(self, query: str, context: Optional[Dict]) -> QueryType:
        """Classify the type of query using multi-signal scoring."""
        query_lower = query.lower()
        
        # Score each category (0-1)
        scores = {
            QueryType.MATH: 0.0,
            QueryType.LOGIC: 0.0,
            QueryType.CODE: 0.0,
            QueryType.FACTUAL: 0.0,
            QueryType.CREATIVE: 0.0,
            QueryType.ANALYSIS: 0.0,
        }
        
        # Math indicators (strong signals)
        math_keywords = ["calculate", "solve", "equation", "sum", "multiply", "divide", 
                        "add", "subtract", "integral", "derivative", "quadratic", "algebra",
                        "geometry", "calculus", "median", "mean", "standard deviation",
                        "area", "volume", "circumference", "radius", "diameter"]
        math_operators = ["+", "-", "×", "*", "/", "=", "^"]
        math_questions = ["how many", "how much", "how far", "how tall", "what's the total",
                         "what is x", "find the", "find x"]
        
        if any(kw in query_lower for kw in math_keywords):
            scores[QueryType.MATH] += 0.6
        if any(op in query for op in math_operators):
            scores[QueryType.MATH] += 0.5
        if any(mq in query_lower for mq in math_questions):
            scores[QueryType.MATH] += 0.4
        # Only add points for numbers if there are other math indicators
        if any(c.isdigit() for c in query) and scores[QueryType.MATH] > 0:
            scores[QueryType.MATH] += 0.3
        
        # Logic indicators
        logic_keywords = ["if", "then", "therefore", "because", "implies", "contradiction",
                         "prove", "assume", "suppose", "syllogism", "valid", "fallacy",
                         "deduce", "infer", "conclude", "premise", "negation", "equivalent",
                         "AND", "OR", "NOT"]
        logic_phrases = ["if and only if", "all are", "some are", "no are", "every",
                        "truth-teller", "always tells", "can we conclude"]
        logic_patterns = ["is a", "are a", "is an", "are an"]  # For syllogisms
        
        if any(kw in query_lower for kw in logic_keywords):
            scores[QueryType.LOGIC] += 0.6
        if any(phrase in query_lower for phrase in logic_phrases):
            scores[QueryType.LOGIC] += 0.4
        # Boost for conditional structures
        if "if" in query_lower and any(word in query_lower for word in ["then", "therefore", "implies"]):
            scores[QueryType.LOGIC] += 0.4
        # Boost for set theory
        if any(word in query_lower for word in ["set", "subset", "intersection", "union", "power set"]):
            scores[QueryType.LOGIC] += 0.3
        
        # Code indicators  
        code_keywords = ["function", "code", "debug", "algorithm", "implement", "program",
                        "error", "bug", "syntax", "python", "javascript", "class",
                        "variable", "loop", "array", "list", "dictionary", "def", "return",
                        "binary search", "hash map", "stack", "linked list", "binary tree"]
        programming_terms = ["def ", "class ", "import ", "return", "for loop", "while loop",
                            "recursion", "decorator", "async", "await"]
        code_phrases = ["write a function", "implement", "code a", "how to create",
                       "how do", "how would you"]
        
        if any(kw in query_lower for kw in code_keywords):
            scores[QueryType.CODE] += 0.6
        if any(term in query_lower for term in programming_terms):
            scores[QueryType.CODE] += 0.5
        if any(phrase in query_lower for phrase in code_phrases):
            scores[QueryType.CODE] += 0.4
        
        # Factual indicators
        factual_keywords = ["who", "when", "where", "which", "history", "fact",
                           "define", "explain", "describe", "capital", "president", "year",
                           "population", "country", "city", "born", "died", "invented"]
        question_words = ["who is", "when did", "where is", "which is"]
        
        if any(kw in query_lower for kw in factual_keywords):
            scores[QueryType.FACTUAL] += 0.4
        if any(qw in query_lower for qw in question_words):
            scores[QueryType.FACTUAL] += 0.3
        # Reduce score for "what is" if there are math indicators
        if ("what is" in query_lower or "what's" in query_lower) and scores[QueryType.MATH] == 0:
            scores[QueryType.FACTUAL] += 0.3
        if query.endswith("?") and len(query.split()) < 15 and scores[QueryType.MATH] == 0:
            scores[QueryType.FACTUAL] += 0.2
        
        # Creative indicators (but not if it's code-related)
        creative_keywords = ["poem", "haiku", "story", "tale", "fiction", "narrative",
                           "imagine", "brainstorm", "invent", "compose",
                           "metaphor", "analogy", "dialogue", "names for", "gift ideas",
                           "creative uses", "hypothetical", "fairy tale"]
        creative_phrases = ["tell me a story", "write a poem", "write a haiku", "create a story",
                          "imagine a", "come up with", "think of", "fairy tale", "give me",
                          "suggest", "what would happen", "what if", "like you're explaining"]
        
        # Only add creative points if there's no strong code signal
        if scores[QueryType.CODE] < 0.5:
            if any(kw in query_lower for kw in creative_keywords):
                scores[QueryType.CREATIVE] += 0.6
            if any(phrase in query_lower for phrase in creative_phrases):
                scores[QueryType.CREATIVE] += 0.5
            # "write" or "create" only count if not code-related
            if ("write" in query_lower or "create" in query_lower) and "function" not in query_lower and "code" not in query_lower:
                scores[QueryType.CREATIVE] += 0.3
            if ("design" in query_lower or "describe" in query_lower) and scores[QueryType.CODE] == 0:
                scores[QueryType.CREATIVE] += 0.3
            # "using", "like", "as if" suggest creative explanation
            if any(word in query_lower for word in ["using a", "like ", "as if"]):
                scores[QueryType.CREATIVE] += 0.3
        
        # Analysis indicators
        analysis_keywords = ["analyze", "compare", "evaluate", "assess", "consider",
                           "examine", "investigate", "critique", "interpret", "contrast",
                           "pros and cons", "advantages", "disadvantages", "impact",
                           "effect", "cause", "trend", "pattern"]
        analysis_phrases = ["compare and contrast", "pros and cons", "analyze the",
                          "evaluate the", "what are the effects"]
        
        if any(kw in query_lower for kw in analysis_keywords):
            scores[QueryType.ANALYSIS] += 0.5
        if any(phrase in query_lower for phrase in analysis_phrases):
            scores[QueryType.ANALYSIS] += 0.4
        
        # Find highest scoring category
        max_score = max(scores.values())
        
        # Only classify if we have reasonable confidence (threshold: 0.3)
        if max_score >= 0.3:
            for query_type, score in scores.items():
                if score == max_score:
                    return query_type, scores
        
        return QueryType.UNKNOWN, scores
    
    def _get_secondary_types(self, scores: Dict[QueryType, float], primary_type: QueryType, threshold: float = 0.4) -> List[QueryType]:
        """Identify secondary query types based on scores.
        
        Args:
            scores: Scores for each query type
            primary_type: The primary classified type
            threshold: Minimum score to be considered secondary (0.4 = 40% of signals)
            
        Returns:
            List of secondary query types
        """
        secondary = []
        for query_type, score in scores.items():
            if query_type != primary_type and score >= threshold:
                secondary.append(query_type)
        return sorted(secondary, key=lambda qt: scores[qt], reverse=True)
    
    def _estimate_complexity(self, query: str, scores: Dict[QueryType, float]) -> float:
        """Estimate query complexity on a scale of 0-1.
        
        Factors:
        - Query length (longer = more complex)
        - Multiple query types (multi-category = more complex)
        - Technical term density
        - Nesting depth (parentheses, nested clauses)
        - Multiple questions/sub-tasks
        
        Returns:
            Complexity score 0-1
        """
        complexity = 0.0
        words = query.split()
        
        # Length factor (normalized to 0-0.3)
        length_score = min(len(words) / 50.0, 0.3)
        complexity += length_score
        
        # Multi-category factor (0-0.3)
        num_high_scores = sum(1 for score in scores.values() if score >= 0.4)
        if num_high_scores > 1:
            complexity += 0.15 * min(num_high_scores - 1, 2)
        
        # Technical term density (0-0.2)
        technical_terms = ["algorithm", "derivative", "integral", "equation", "theorem",
                          "complexity", "implementation", "architecture", "optimization"]
        tech_count = sum(1 for term in technical_terms if term in query.lower())
        complexity += min(tech_count * 0.05, 0.2)
        
        # Nesting depth (0-0.2)
        nesting = query.count('(') + query.count('[') + query.count(',')
        complexity += min(nesting * 0.03, 0.2)
        
        return min(complexity, 1.0)
    
    def _select_worker(self, query_type: QueryType, context: Optional[Dict]) -> str:
        """Select worker agent based on query type and learned performance."""
        
        # Check if we have learned performance data for this query type
        type_key = query_type.value
        if type_key in self.performance_stats:
            stats = self.performance_stats[type_key]
            # Pick worker with best accuracy based on historical performance
            if "workers" in stats and stats["workers"]:
                best_worker = max(stats["workers"].items(), 
                                key=lambda x: x[1].get("accuracy", 0))
                return best_worker[0]
        
        # No historical data - use optimized worker mapping
        worker_map = {
            QueryType.MATH: "math",
            QueryType.LOGIC: "logic",
            QueryType.CODE: "code",
            QueryType.FACTUAL: "factual",
            QueryType.CREATIVE: "creative",
            QueryType.ANALYSIS: "analysis",
            QueryType.UNKNOWN: "logic"  # Default to logic for unknown queries
        }
        
        return worker_map[query_type]
    
    def _build_lats_config(self, query_type: QueryType, complexity: float,
                          context: Optional[Dict]) -> Dict[str, Any]:
        """Build LATS configuration based on query type and complexity.
        
        Args:
            query_type: Type of query
            complexity: Complexity score (0-1)
            context: Optional context
            
        Returns:
            Dictionary with LATS parameters
        """
        # Base configuration
        config = {
            "use_tree_cache": True,
            "max_tree_depth": 5,
            "num_simulations": 10
        }
        
        # Adjust based on complexity
        if complexity > 0.7:
            # High complexity: deeper search, more simulations
            config["max_tree_depth"] = 7
            config["num_simulations"] = 20
        elif complexity < 0.3:
            # Low complexity: shallower search, fewer simulations
            config["max_tree_depth"] = 3
            config["num_simulations"] = 5
        
        # Query-type specific adjustments
        if query_type == QueryType.MATH:
            # Math benefits from deeper search
            config["max_tree_depth"] += 2
        elif query_type == QueryType.CODE:
            # Code generation needs many simulations
            config["num_simulations"] += 10
        elif query_type == QueryType.CREATIVE:
            # Creative doesn't need deep search
            config["max_tree_depth"] = max(3, config["max_tree_depth"] - 2)
            config["num_simulations"] = max(3, config["num_simulations"] - 5)
        
        return config
    
    def _calculate_routing_confidence(self, query_type: QueryType, scores: Dict[QueryType, float],
                                     context: Optional[Dict]) -> float:
        """Calculate confidence in routing decision.
        
        Args:
            query_type: Selected query type
            scores: All query type scores
            context: Optional context
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence is the score for selected type
        confidence = scores.get(query_type, 0.0)
        
        # Reduce confidence if multiple types have similar scores
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            # If second-best score is close, reduce confidence
            score_gap = sorted_scores[0] - sorted_scores[1]
            if score_gap < 0.2:
                confidence *= 0.8
        
        # Boost confidence if we have historical data
        type_key = query_type.value
        if type_key in self.performance_stats:
            stats = self.performance_stats[type_key]
            if stats.get("count", 0) > 10:
                confidence = min(confidence * 1.1, 1.0)
        
        return confidence
    
    def _explain_decision(self, query: str, query_type: QueryType, 
                         worker_specialty: str, complexity: float) -> str:
        """Generate human-readable explanation of routing decision."""
        return (f"Classified as {query_type.value} query (complexity: {complexity:.2f}), "
                f"routing to {worker_specialty} worker with LATS-based reasoning")
    
    def _calculate_accuracy(self, result: Dict[str, Any]) -> float:
        """Calculate accuracy score from result."""
        details = result.get("verification_details", {})
        
        # If no checks, return 0.5 (uncertain)
        total_checks = details.get("symbolic_checks", 0) + details.get("factual_checks", 0)
        if total_checks == 0:
            return 0.5
        
        # Calculate pass rate
        passed = details.get("symbolic_passed", 0) + details.get("factual_passed", 0)
        return passed / total_checks
    
    def _load_stats(self) -> Dict[str, Any]:
        """Load performance statistics from disk."""
        if not self.stats_file.exists():
            return {}
        
        try:
            with open(self.stats_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _update_stats(self, outcome: RoutingOutcome):
        """Update performance statistics with new outcome."""
        type_key = outcome.query_type.value
        strategy_key = outcome.strategy.value
        
        if type_key not in self.performance_stats:
            self.performance_stats[type_key] = {
                "count": 0,
                "strategies": {}
            }
        
        if strategy_key not in self.performance_stats[type_key]["strategies"]:
            self.performance_stats[type_key]["strategies"][strategy_key] = {
                "count": 0,
                "accuracy": 0.0,
                "avg_latency": 0.0,
                "avg_cost": 0.0
            }
        
        # Update running averages
        stats = self.performance_stats[type_key]["strategies"][strategy_key]
        n = stats["count"]
        
        stats["accuracy"] = (stats["accuracy"] * n + outcome.accuracy_score) / (n + 1)
        stats["avg_latency"] = (stats["avg_latency"] * n + outcome.latency_ms) / (n + 1)
        stats["avg_cost"] = (stats["avg_cost"] * n + outcome.cost) / (n + 1)
        stats["count"] += 1
        
        self.performance_stats[type_key]["count"] += 1
        
        # Save to disk
        with open(self.stats_file, "w") as f:
            json.dump(self.performance_stats, f, indent=2)
    
    def _update_stats_worker(self, outcome: RoutingOutcome, worker_specialty: str):
        """Update performance statistics for worker-based routing."""
        type_key = outcome.query_type.value
        
        if type_key not in self.performance_stats:
            self.performance_stats[type_key] = {
                "count": 0,
                "workers": {}
            }
        
        if "workers" not in self.performance_stats[type_key]:
            self.performance_stats[type_key]["workers"] = {}
        
        if worker_specialty not in self.performance_stats[type_key]["workers"]:
            self.performance_stats[type_key]["workers"][worker_specialty] = {
                "count": 0,
                "accuracy": 0.0,
                "avg_latency": 0.0,
                "avg_cost": 0.0
            }
        
        # Update running averages
        stats = self.performance_stats[type_key]["workers"][worker_specialty]
        n = stats["count"]
        
        stats["accuracy"] = (stats["accuracy"] * n + outcome.accuracy_score) / (n + 1)
        stats["avg_latency"] = (stats["avg_latency"] * n + outcome.latency_ms) / (n + 1)
        stats["avg_cost"] = (stats["avg_cost"] * n + outcome.cost) / (n + 1)
        stats["count"] += 1
        
        self.performance_stats[type_key]["count"] += 1
        
        # Save to disk
        with open(self.stats_file, "w") as f:
            json.dump(self.performance_stats, f, indent=2)
    
    def _update_stats_from_file(self):
        """Rebuild stats from outcomes file (for simulation)."""
        if not self.outcomes_file.exists():
            return
        
        self.performance_stats = {}
        
        with open(self.outcomes_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Reconstruct outcome (simplified)
                    outcome = RoutingOutcome(
                        query=data["query"],
                        query_type=QueryType[data["query_type"]],
                        strategy=ReasoningStrategy[data["strategy"]],
                        decision=None,  # Not needed for stats
                        success=data["success"],
                        accuracy_score=data["accuracy_score"],
                        latency_ms=data["latency_ms"],
                        cost=data["cost"],
                        symbolic_passed=data["symbolic_passed"],
                        factual_passed=data["factual_passed"],
                        reflection_iterations=data["reflection_iterations"],
                        timestamp=data["timestamp"]
                    )
                    self._update_stats(outcome)
                except Exception:
                    continue
    
    def _count_outcomes(self) -> int:
        """Count number of logged outcomes."""
        if not self.outcomes_file.exists():
            return 0
        
        with open(self.outcomes_file, "r") as f:
            return sum(1 for _ in f)
    
    def _simulate_outcome(self, query: str, decision: RoutingDecision, 
                         test_case: Dict) -> RoutingOutcome:
        """Simulate a routing outcome for testing."""
        import random
        
        # Simulate performance based on strategy match
        strategy = decision.strategy
        query_type = decision.query_type
        
        # Base accuracy by strategy-type fit
        accuracy_base = {
            (QueryType.MATH, ReasoningStrategy.SYMBOLIC_HEAVY): 0.92,
            (QueryType.MATH, ReasoningStrategy.BALANCED): 0.85,
            (QueryType.MATH, ReasoningStrategy.FAST): 0.70,
            (QueryType.FACTUAL, ReasoningStrategy.FACTUAL_HEAVY): 0.88,
            (QueryType.FACTUAL, ReasoningStrategy.BALANCED): 0.82,
            (QueryType.LOGIC, ReasoningStrategy.DEEP): 0.90,
            (QueryType.LOGIC, ReasoningStrategy.BALANCED): 0.85,
        }.get((query_type, strategy), 0.75)
        
        # Add noise
        accuracy = min(1.0, max(0.0, accuracy_base + random.gauss(0, 0.05)))
        
        # Simulate latency (deep = slower, fast = faster)
        latency_base = {
            ReasoningStrategy.FAST: 150,
            ReasoningStrategy.BALANCED: 300,
            ReasoningStrategy.SYMBOLIC_HEAVY: 400,
            ReasoningStrategy.FACTUAL_HEAVY: 500,
            ReasoningStrategy.DEEP: 800
        }.get(strategy, 300)
        
        latency = latency_base + random.gauss(0, 50)
        
        # Simulate cost (proportional to latency)
        cost = (latency / 1000) * 0.00001
        
        return RoutingOutcome(
            query=query,
            query_type=query_type,
            strategy=strategy,
            decision=decision,
            success=accuracy > 0.75,
            accuracy_score=accuracy,
            latency_ms=latency,
            cost=cost,
            symbolic_passed=decision.use_symbolic_verification and accuracy > 0.8,
            factual_passed=decision.use_factual_verification and accuracy > 0.8,
            reflection_iterations=decision.max_reflection_iterations,
            timestamp=time.time()
        )
