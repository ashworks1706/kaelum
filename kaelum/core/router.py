"""Adaptive routing and strategy selection for reasoning queries.

The Router is Kaelum's "brain" - it learns which reasoning strategies work best
for different types of queries by analyzing past performance and simulations.

Phase 1: Rule-based routing with learning
Phase 2: Small neural policy model (1-2B parameters)
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

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
    """Decision made by the router."""
    query_type: QueryType
    strategy: ReasoningStrategy
    max_reflection_iterations: int
    use_symbolic_verification: bool
    use_factual_verification: bool
    confidence_threshold: float
    reasoning: str = ""
    # New fields for Phase 1.5
    secondary_types: List[QueryType] = None  # Multi-category detection
    complexity_score: float = 0.0  # 0-1, estimated query complexity
    
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
    """Adaptive router that learns optimal reasoning strategies.
    
    Phase 1: Rule-based heuristics + outcome tracking
    Phase 2: Neural policy model trained on collected data
    """
    
    def __init__(self, learning_enabled: bool = True, data_dir: str = ".kaelum/routing"):
        self.learning_enabled = learning_enabled
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.outcomes_file = self.data_dir / "outcomes.jsonl"
        self.stats_file = self.data_dir / "stats.json"
        self.decisions_log = self.data_dir / "routing_decisions.log"
        
        # Set up file handler for routing decisions log
        fh = logging.FileHandler(self.decisions_log)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(fh)
        
        # Load historical performance stats
        self.performance_stats = self._load_stats()
        
        logger.info("=" * 60)
        logger.info("Router initialized")
        logger.info(f"Learning enabled: {learning_enabled}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info("=" * 60)
        
    def route(self, query: str, context: Optional[Dict] = None) -> RoutingDecision:
        """Route a query to the optimal reasoning strategy.
        
        Args:
            query: The input query to route
            context: Optional context (previous results, user preferences, etc.)
            
        Returns:
            RoutingDecision with strategy and configuration
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
        
        # Step 2: Select strategy based on query type + learned performance
        strategy = self._select_strategy(query_type, context)
        logger.info(f"  Strategy: {strategy.value}")
        
        # Step 3: Configure reasoning parameters for this strategy
        config = self._build_config(query_type, strategy, context)
        logger.info(f"  Config: reflection={config['max_reflection_iterations']}, "
                   f"symbolic={config['use_symbolic_verification']}, "
                   f"factual={config['use_factual_verification']}")
        
        # Step 4: Generate reasoning for decision (for observability)
        reasoning = self._explain_decision(query, query_type, strategy, config)
        logger.info(f"  Reasoning: {reasoning}")
        
        routing_time = (time.time() - start_time) * 1000
        logger.info(f"  Routing time: {routing_time:.2f}ms")
        logger.info("-" * 60)
        
        return RoutingDecision(
            query_type=query_type,
            strategy=strategy,
            **config,
            reasoning=reasoning,
            secondary_types=secondary_types,
            complexity_score=complexity
        )
    
    def record_outcome(self, decision: RoutingDecision, result: Dict[str, Any]):
        """Record the outcome of a routing decision for learning.
        
        Args:
            decision: The routing decision that was made
            result: The result from the orchestrator (includes metrics, verification)
        """
        if not self.learning_enabled:
            return
        
        outcome = RoutingOutcome(
            query=result.get("query", ""),
            query_type=decision.query_type,
            strategy=decision.strategy,
            decision=decision,
            
            success=len(result.get("verification_errors", [])) == 0,
            accuracy_score=self._calculate_accuracy(result),
            latency_ms=result.get("metrics", {}).get("total_time_ms", 0),
            cost=result.get("metrics", {}).get("local_cost", 0),
            
            symbolic_passed=result.get("verification_details", {}).get("symbolic_checks", 0) == 
                           result.get("verification_details", {}).get("symbolic_passed", 0),
            factual_passed=result.get("verification_details", {}).get("factual_checks", 0) == 
                          result.get("verification_details", {}).get("factual_passed", 0),
            reflection_iterations=result.get("metrics", {}).get("reflection_time_ms", 0) > 0,
            
            timestamp=time.time()
        )
        
        logger.info("=" * 60)
        logger.info("ROUTING OUTCOME")
        logger.info(f"  Success: {outcome.success}")
        logger.info(f"  Accuracy: {outcome.accuracy_score:.2f}")
        logger.info(f"  Latency: {outcome.latency_ms:.2f}ms")
        logger.info(f"  Cost: ${outcome.cost:.6f}")
        logger.info(f"  Symbolic verification: {'PASS' if outcome.symbolic_passed else 'FAIL'}")
        logger.info(f"  Factual verification: {'PASS' if outcome.factual_passed else 'FAIL'}")
        logger.info("=" * 60)
        
        # Append to outcomes log
        with open(self.outcomes_file, "a") as f:
            f.write(json.dumps(asdict(outcome), default=str) + "\n")
        
        # Update performance stats
        self._update_stats(outcome)
    
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
    
    def _select_strategy(self, query_type: QueryType, context: Optional[Dict]) -> ReasoningStrategy:
        """Select reasoning strategy based on query type and learned performance."""
        
        # Check if we have learned performance data for this query type
        type_key = query_type.value
        if type_key in self.performance_stats:
            stats = self.performance_stats[type_key]
            # Pick strategy with best accuracy based on historical performance
            best_strategy = max(stats["strategies"].items(), 
                              key=lambda x: x[1].get("accuracy", 0))
            # Convert lowercase key to uppercase enum
            return ReasoningStrategy[best_strategy[0].upper()]
        
        # No historical data - use optimized strategy mapping
        strategy_map = {
            QueryType.MATH: ReasoningStrategy.SYMBOLIC_HEAVY,
            QueryType.LOGIC: ReasoningStrategy.BALANCED,
            QueryType.CODE: ReasoningStrategy.DEEP,
            QueryType.FACTUAL: ReasoningStrategy.FACTUAL_HEAVY,
            QueryType.CREATIVE: ReasoningStrategy.FAST,
            QueryType.ANALYSIS: ReasoningStrategy.BALANCED,
            QueryType.UNKNOWN: ReasoningStrategy.BALANCED
        }
        
        return strategy_map[query_type]
    
    def _build_config(self, query_type: QueryType, strategy: ReasoningStrategy, 
                     context: Optional[Dict]) -> Dict[str, Any]:
        """Build reasoning configuration for the selected strategy."""
        
        configs = {
            ReasoningStrategy.SYMBOLIC_HEAVY: {
                "max_reflection_iterations": 2,
                "use_symbolic_verification": True,
                "use_factual_verification": False,
                "confidence_threshold": 0.85
            },
            ReasoningStrategy.FACTUAL_HEAVY: {
                "max_reflection_iterations": 1,
                "use_symbolic_verification": False,
                "use_factual_verification": True,
                "confidence_threshold": 0.80
            },
            ReasoningStrategy.BALANCED: {
                "max_reflection_iterations": 2,
                "use_symbolic_verification": True,
                "use_factual_verification": True,
                "confidence_threshold": 0.75
            },
            ReasoningStrategy.FAST: {
                "max_reflection_iterations": 0,
                "use_symbolic_verification": True,
                "use_factual_verification": False,
                "confidence_threshold": 0.70
            },
            ReasoningStrategy.DEEP: {
                "max_reflection_iterations": 3,
                "use_symbolic_verification": True,
                "use_factual_verification": True,
                "confidence_threshold": 0.90
            }
        }
        
        return configs.get(strategy, configs[ReasoningStrategy.BALANCED])
    
    def _explain_decision(self, query: str, query_type: QueryType, 
                         strategy: ReasoningStrategy, config: Dict) -> str:
        """Generate human-readable explanation of routing decision."""
        return (f"Classified as {query_type.value} query, "
                f"selected {strategy.value} strategy "
                f"(reflection={config['max_reflection_iterations']}, "
                f"symbolic={config['use_symbolic_verification']}, "
                f"factual={config['use_factual_verification']})")
    
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
