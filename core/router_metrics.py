"""Router metrics collection and analysis.

Tracks routing performance across queries to identify patterns and improvements.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy."""
    strategy_name: str
    total_queries: int
    success_count: int
    failure_count: int
    total_latency_ms: float
    total_cost: float
    accuracy_scores: List[float]
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_queries == 0:
            return 0.0
        return (self.success_count / self.total_queries) * 100
    
    @property
    def average_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries
    
    @property
    def average_cost(self) -> float:
        """Average cost per query."""
        if self.total_queries == 0:
            return 0.0
        return self.total_cost / self.total_queries
    
    @property
    def average_accuracy(self) -> float:
        """Average accuracy score (0-1)."""
        if not self.accuracy_scores:
            return 0.0
        return sum(self.accuracy_scores) / len(self.accuracy_scores)


@dataclass
class QueryTypeMetrics:
    """Metrics for a single query type."""
    query_type: str
    total_queries: int
    strategies_used: Dict[str, int]
    average_accuracy: float
    average_latency_ms: float
    
    @property
    def most_used_strategy(self) -> str:
        """Most frequently used strategy for this query type."""
        if not self.strategies_used:
            return "unknown"
        return max(self.strategies_used.items(), key=lambda x: x[1])[0]


class RouterMetricsCollector:
    """Collects and analyzes routing metrics."""
    
    def __init__(self, data_dir: str = ".kaelum/routing"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.outcomes_file = self.data_dir / "outcomes.jsonl"
        self.metrics_file = self.data_dir / "metrics.json"
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from outcomes log."""
        if not self.outcomes_file.exists():
            return self._empty_metrics()
        
        # Read all outcomes
        outcomes = []
        with open(self.outcomes_file, "r") as f:
            for line in f:
                try:
                    outcomes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        if not outcomes:
            return self._empty_metrics()
        
        # Aggregate by strategy
        strategy_data = defaultdict(lambda: {
            "total": 0,
            "success": 0,
            "failure": 0,
            "latency": 0.0,
            "cost": 0.0,
            "accuracies": []
        })
        
        # Aggregate by query type
        query_type_data = defaultdict(lambda: {
            "total": 0,
            "strategies": defaultdict(int),
            "accuracies": [],
            "latencies": []
        })
        
        for outcome in outcomes:
            strategy = outcome.get("strategy", "unknown")
            query_type = outcome.get("query_type", "unknown")
            success = outcome.get("success", False)
            accuracy = outcome.get("accuracy_score", 0.0)
            latency = outcome.get("latency_ms", 0.0)
            cost = outcome.get("cost", 0.0)
            
            # Update strategy metrics
            strategy_data[strategy]["total"] += 1
            if success:
                strategy_data[strategy]["success"] += 1
            else:
                strategy_data[strategy]["failure"] += 1
            strategy_data[strategy]["latency"] += latency
            strategy_data[strategy]["cost"] += cost
            strategy_data[strategy]["accuracies"].append(accuracy)
            
            # Update query type metrics
            query_type_data[query_type]["total"] += 1
            query_type_data[query_type]["strategies"][strategy] += 1
            query_type_data[query_type]["accuracies"].append(accuracy)
            query_type_data[query_type]["latencies"].append(latency)
        
        # Build strategy metrics
        strategy_metrics = {}
        for strategy, data in strategy_data.items():
            strategy_metrics[strategy] = StrategyMetrics(
                strategy_name=strategy,
                total_queries=data["total"],
                success_count=data["success"],
                failure_count=data["failure"],
                total_latency_ms=data["latency"],
                total_cost=data["cost"],
                accuracy_scores=data["accuracies"]
            )
        
        # Build query type metrics
        query_type_metrics = {}
        for qtype, data in query_type_data.items():
            query_type_metrics[qtype] = QueryTypeMetrics(
                query_type=qtype,
                total_queries=data["total"],
                strategies_used=dict(data["strategies"]),
                average_accuracy=sum(data["accuracies"]) / len(data["accuracies"]) if data["accuracies"] else 0.0,
                average_latency_ms=sum(data["latencies"]) / len(data["latencies"]) if data["latencies"] else 0.0
            )
        
        # Calculate overall metrics
        total_queries = sum(data["total"] for data in strategy_data.values())
        total_success = sum(data["success"] for data in strategy_data.values())
        total_latency = sum(data["latency"] for data in strategy_data.values())
        total_cost = sum(data["cost"] for data in strategy_data.values())
        all_accuracies = [acc for data in strategy_data.values() for acc in data["accuracies"]]
        
        metrics = {
            "timestamp": time.time(),
            "total_queries": total_queries,
            "overall_success_rate": (total_success / total_queries * 100) if total_queries > 0 else 0.0,
            "overall_accuracy": sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0,
            "average_latency_ms": total_latency / total_queries if total_queries > 0 else 0.0,
            "total_cost": total_cost,
            "by_strategy": {
                name: {
                    "total_queries": m.total_queries,
                    "success_rate": m.success_rate,
                    "average_latency_ms": m.average_latency_ms,
                    "average_cost": m.average_cost,
                    "average_accuracy": m.average_accuracy
                }
                for name, m in strategy_metrics.items()
            },
            "by_query_type": {
                name: {
                    "total_queries": m.total_queries,
                    "most_used_strategy": m.most_used_strategy,
                    "average_accuracy": m.average_accuracy,
                    "average_latency_ms": m.average_latency_ms,
                    "strategies_distribution": m.strategies_used
                }
                for name, m in query_type_metrics.items()
            }
        }
        
        # Save to file
        with open(self.metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "timestamp": time.time(),
            "total_queries": 0,
            "overall_success_rate": 0.0,
            "overall_accuracy": 0.0,
            "average_latency_ms": 0.0,
            "total_cost": 0.0,
            "by_strategy": {},
            "by_query_type": {}
        }
    
    def get_top_strategies(self, n: int = 3) -> List[str]:
        """Get top N performing strategies by success rate."""
        metrics = self.collect_metrics()
        strategies = metrics.get("by_strategy", {})
        
        if not strategies:
            return []
        
        # Sort by success rate
        sorted_strategies = sorted(
            strategies.items(),
            key=lambda x: x[1]["success_rate"],
            reverse=True
        )
        
        return [name for name, _ in sorted_strategies[:n]]
    
    def get_strategy_recommendation(self, query_type: str) -> str:
        """Get recommended strategy for a query type based on historical performance."""
        metrics = self.collect_metrics()
        query_types = metrics.get("by_query_type", {})
        
        if query_type not in query_types:
            return "balanced"  # Default for unknown query types
        
        return query_types[query_type]["most_used_strategy"]
    
    def format_summary(self, metrics: Optional[Dict] = None) -> str:
        """Format metrics as a human-readable summary."""
        if metrics is None:
            metrics = self.collect_metrics()
        
        if metrics["total_queries"] == 0:
            return "No routing data available yet."
        
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("ROUTER PERFORMANCE SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Total Queries: {metrics['total_queries']}")
        lines.append(f"Overall Success Rate: {metrics['overall_success_rate']:.1f}%")
        lines.append(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}")
        lines.append(f"Average Latency: {metrics['average_latency_ms']:.1f}ms")
        lines.append(f"Total Cost: ${metrics['total_cost']:.6f}")
        
        lines.append("\n" + "-" * 60)
        lines.append("BY STRATEGY")
        lines.append("-" * 60)
        
        for strategy, data in sorted(metrics["by_strategy"].items(), 
                                     key=lambda x: x[1]["total_queries"], 
                                     reverse=True):
            lines.append(f"\n{strategy.upper()}:")
            lines.append(f"  Queries: {data['total_queries']}")
            lines.append(f"  Success Rate: {data['success_rate']:.1f}%")
            lines.append(f"  Avg Accuracy: {data['average_accuracy']:.2f}")
            lines.append(f"  Avg Latency: {data['average_latency_ms']:.1f}ms")
            lines.append(f"  Avg Cost: ${data['average_cost']:.6f}")
        
        lines.append("\n" + "-" * 60)
        lines.append("BY QUERY TYPE")
        lines.append("-" * 60)
        
        for qtype, data in sorted(metrics["by_query_type"].items(),
                                 key=lambda x: x[1]["total_queries"],
                                 reverse=True):
            lines.append(f"\n{qtype.upper()}:")
            lines.append(f"  Queries: {data['total_queries']}")
            lines.append(f"  Preferred Strategy: {data['most_used_strategy']}")
            lines.append(f"  Avg Accuracy: {data['average_accuracy']:.2f}")
            lines.append(f"  Avg Latency: {data['average_latency_ms']:.1f}ms")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
