"""Adaptive routing and strategy selection for reasoning queries.

The Router is Kaelum's "brain" - it learns which reasoning strategies work best
for different types of queries by analyzing past performance and simulations.

Phase 1: Rule-based routing with learning
Phase 2: Small neural policy model (1-2B parameters)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


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
    reasoning: str  # Why this decision was made


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
        
        # Load historical performance stats
        self.performance_stats = self._load_stats()
        
    def route(self, query: str, context: Optional[Dict] = None) -> RoutingDecision:
        """Route a query to the optimal reasoning strategy.
        
        Args:
            query: The input query to route
            context: Optional context (previous results, user preferences, etc.)
            
        Returns:
            RoutingDecision with strategy and configuration
        """
        # Step 1: Classify query type
        query_type = self._classify_query(query, context)
        
        # Step 2: Select strategy based on query type + learned performance
        strategy = self._select_strategy(query_type, context)
        
        # Step 3: Configure reasoning parameters for this strategy
        config = self._build_config(query_type, strategy, context)
        
        # Step 4: Generate reasoning for decision (for observability)
        reasoning = self._explain_decision(query, query_type, strategy, config)
        
        return RoutingDecision(
            query_type=query_type,
            strategy=strategy,
            **config,
            reasoning=reasoning
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
        """Classify the type of query (Phase 1: rule-based)."""
        query_lower = query.lower()
        
        # Math indicators
        math_keywords = ["calculate", "solve", "equation", "sum", "multiply", "divide", 
                        "add", "subtract", "+", "-", "×", "=", "integral", "derivative"]
        if any(kw in query_lower for kw in math_keywords) or any(c in query for c in "0123456789"):
            return QueryType.MATH
        
        # Logic indicators
        logic_keywords = ["if", "then", "therefore", "because", "implies", "contradiction",
                         "prove", "assume", "suppose"]
        if any(kw in query_lower for kw in logic_keywords):
            return QueryType.LOGIC
        
        # Code indicators
        code_keywords = ["function", "code", "debug", "algorithm", "implement", "program",
                        "error", "bug", "syntax"]
        if any(kw in query_lower for kw in code_keywords):
            return QueryType.CODE
        
        # Factual indicators
        factual_keywords = ["who", "what", "when", "where", "which", "history", "fact",
                           "define", "explain", "describe"]
        if any(kw in query_lower for kw in factual_keywords):
            return QueryType.FACTUAL
        
        # Analysis indicators
        analysis_keywords = ["analyze", "compare", "evaluate", "assess", "consider",
                           "examine", "investigate"]
        if any(kw in query_lower for kw in analysis_keywords):
            return QueryType.ANALYSIS
        
        return QueryType.UNKNOWN
    
    def _select_strategy(self, query_type: QueryType, context: Optional[Dict]) -> ReasoningStrategy:
        """Select reasoning strategy based on query type and learned performance."""
        
        # Check if we have learned performance data for this query type
        type_key = query_type.value
        if type_key in self.performance_stats:
            stats = self.performance_stats[type_key]
            # Pick strategy with best accuracy (Phase 1 simple heuristic)
            best_strategy = max(stats["strategies"].items(), 
                              key=lambda x: x[1].get("accuracy", 0))
            # Convert lowercase key to uppercase enum
            return ReasoningStrategy[best_strategy[0].upper()]
        
        # Fallback: rule-based strategy selection
        strategy_map = {
            QueryType.MATH: ReasoningStrategy.SYMBOLIC_HEAVY,
            QueryType.LOGIC: ReasoningStrategy.BALANCED,
            QueryType.CODE: ReasoningStrategy.DEEP,
            QueryType.FACTUAL: ReasoningStrategy.FACTUAL_HEAVY,
            QueryType.CREATIVE: ReasoningStrategy.FAST,
            QueryType.ANALYSIS: ReasoningStrategy.BALANCED,
            QueryType.UNKNOWN: ReasoningStrategy.BALANCED
        }
        
        return strategy_map.get(query_type, ReasoningStrategy.BALANCED)
    
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
