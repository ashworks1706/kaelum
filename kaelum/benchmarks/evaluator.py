"""Benchmark evaluator to calculate metrics and generate reports."""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from kaelum.benchmarks.dataset import BenchmarkQuery, BenchmarkDataset, QueryCategory, DifficultyLevel
from kaelum.benchmarks.runner import BenchmarkResult, RunMode


@dataclass
class EvaluationMetrics:
    """Metrics for benchmark evaluation.
    
    Attributes:
        accuracy: Correctness rate (0-1)
        avg_confidence: Average confidence score
        avg_execution_time: Average time in seconds
        verification_rate: Percentage of verified answers
        total_queries: Total number of queries
        speedup: Speedup factor vs baseline (if applicable)
    """
    accuracy: float
    avg_confidence: float
    avg_execution_time: float
    verification_rate: float
    total_queries: int
    speedup: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": round(self.accuracy, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "avg_execution_time": round(self.avg_execution_time, 4),
            "verification_rate": round(self.verification_rate, 3),
            "total_queries": self.total_queries,
            "speedup": round(self.speedup, 2) if self.speedup else None
        }


class BenchmarkEvaluator:
    """Evaluates benchmark results and generates reports."""
    
    def __init__(self, dataset: BenchmarkDataset):
        """Initialize evaluator.
        
        Args:
            dataset: Dataset with expected answers
        """
        self.dataset = dataset
    
    def evaluate_result(
        self,
        result: BenchmarkResult,
        query: BenchmarkQuery
    ) -> bool:
        """Check if a result is correct.
        
        Args:
            result: Result to evaluate
            query: Original query with expected answer
            
        Returns:
            True if correct
        """
        if query.expected_answer is None:
            # No expected answer - use confidence as proxy
            return result.confidence > 0.7
        
        # Normalize both answers for comparison
        expected = self._normalize_answer(query.expected_answer)
        actual = self._normalize_answer(result.answer)
        
        # Check if actual contains expected (allows for extra explanation)
        return expected in actual
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        return answer.lower().strip().replace(" ", "").replace(",", "")
    
    def calculate_metrics(
        self,
        results: List[BenchmarkResult],
        baseline_results: Optional[List[BenchmarkResult]] = None
    ) -> EvaluationMetrics:
        """Calculate metrics for results.
        
        Args:
            results: Results to evaluate
            baseline_results: Optional baseline for speedup calculation
            
        Returns:
            Evaluation metrics
        """
        if not results:
            return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0)
        
        # Calculate accuracy
        correct = 0
        for result in results:
            query = self.dataset.get_by_id(result.query_id)
            if query and self.evaluate_result(result, query):
                correct += 1
        
        accuracy = correct / len(results)
        
        # Calculate other metrics
        avg_confidence = sum(r.confidence for r in results) / len(results)
        avg_time = sum(r.execution_time for r in results) / len(results)
        verified_count = sum(1 for r in results if r.verified)
        verification_rate = verified_count / len(results)
        
        # Calculate speedup if baseline provided
        speedup = None
        if baseline_results:
            baseline_time = sum(r.execution_time for r in baseline_results) / len(baseline_results)
            if baseline_time > 0 and avg_time > 0:
                speedup = baseline_time / avg_time
        
        return EvaluationMetrics(
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            avg_execution_time=avg_time,
            verification_rate=verification_rate,
            total_queries=len(results),
            speedup=speedup
        )
    
    def compare_modes(
        self,
        single_results: List[BenchmarkResult],
        meta_results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Compare single-worker vs meta-reasoner results.
        
        Args:
            single_results: Single worker results
            meta_results: Meta-reasoner results
            
        Returns:
            Comparison report
        """
        single_metrics = self.calculate_metrics(single_results)
        meta_metrics = self.calculate_metrics(meta_results, baseline_results=single_results)
        
        # Calculate improvements
        accuracy_improvement = (meta_metrics.accuracy - single_metrics.accuracy) * 100
        confidence_improvement = (meta_metrics.avg_confidence - single_metrics.avg_confidence) * 100
        
        return {
            "single_worker": single_metrics.to_dict(),
            "meta_reasoner": meta_metrics.to_dict(),
            "improvements": {
                "accuracy_pct": round(accuracy_improvement, 2),
                "confidence_pct": round(confidence_improvement, 2),
                "speedup": meta_metrics.speedup
            },
            "recommendation": self._get_recommendation(single_metrics, meta_metrics)
        }
    
    def _get_recommendation(
        self,
        single: EvaluationMetrics,
        meta: EvaluationMetrics
    ) -> str:
        """Generate recommendation based on metrics."""
        if meta.accuracy > single.accuracy * 1.1:  # 10% improvement
            return "Meta-reasoner shows significant accuracy improvement"
        elif meta.speedup and meta.speedup > 2.0:
            return "Meta-reasoner provides good speedup through parallelism"
        elif meta.verification_rate > single.verification_rate * 1.2:
            return "Meta-reasoner has better verification rate"
        else:
            return "Single worker may be sufficient for this workload"
    
    def analyze_by_category(
        self,
        results: List[BenchmarkResult]
    ) -> Dict[str, EvaluationMetrics]:
        """Analyze results grouped by query category.
        
        Args:
            results: Results to analyze
            
        Returns:
            Metrics per category
        """
        by_category = defaultdict(list)
        
        for result in results:
            category = result.metadata.get("category", "unknown")
            by_category[category].append(result)
        
        return {
            category: self.calculate_metrics(results)
            for category, results in by_category.items()
        }
    
    def analyze_by_difficulty(
        self,
        results: List[BenchmarkResult]
    ) -> Dict[str, EvaluationMetrics]:
        """Analyze results grouped by difficulty level.
        
        Args:
            results: Results to analyze
            
        Returns:
            Metrics per difficulty
        """
        by_difficulty = defaultdict(list)
        
        for result in results:
            difficulty = result.metadata.get("difficulty", "unknown")
            by_difficulty[difficulty].append(result)
        
        return {
            difficulty: self.calculate_metrics(results)
            for difficulty, results in by_difficulty.items()
        }
    
    def generate_report(
        self,
        results: List[BenchmarkResult],
        comparison_results: Optional[List[BenchmarkResult]] = None,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive benchmark report.
        
        Args:
            results: Primary results to report on
            comparison_results: Optional baseline for comparison
            output_file: Optional file to save JSON report
            
        Returns:
            Report dictionary
        """
        overall_metrics = self.calculate_metrics(results, comparison_results)
        
        report = {
            "summary": {
                "dataset": self.dataset.name,
                "total_queries": len(results),
                "metrics": overall_metrics.to_dict()
            },
            "by_category": {
                cat: metrics.to_dict()
                for cat, metrics in self.analyze_by_category(results).items()
            },
            "by_difficulty": {
                diff: metrics.to_dict()
                for diff, metrics in self.analyze_by_difficulty(results).items()
            }
        }
        
        # Add comparison if provided
        if comparison_results:
            # Group by mode
            single_results = [r for r in comparison_results if r.mode == RunMode.SINGLE_WORKER]
            meta_results = [r for r in results if r.mode == RunMode.META_REASONER]
            
            if single_results and meta_results:
                report["comparison"] = self.compare_modes(single_results, meta_results)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self, metrics: EvaluationMetrics, title: str = "Benchmark Results"):
        """Print human-readable summary.
        
        Args:
            metrics: Metrics to display
            title: Report title
        """
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        print(f"Total Queries:       {metrics.total_queries}")
        print(f"Accuracy:            {metrics.accuracy*100:.1f}%")
        print(f"Avg Confidence:      {metrics.avg_confidence*100:.1f}%")
        print(f"Avg Execution Time:  {metrics.avg_execution_time:.4f}s")
        print(f"Verification Rate:   {metrics.verification_rate*100:.1f}%")
        if metrics.speedup:
            print(f"Speedup vs Baseline: {metrics.speedup:.2f}x")
        print(f"{'='*60}\n")
    
    def print_comparison(
        self,
        single_results: List[BenchmarkResult],
        meta_results: List[BenchmarkResult]
    ):
        """Print comparison between single and meta modes.
        
        Args:
            single_results: Single worker results
            meta_results: Meta-reasoner results
        """
        comparison = self.compare_modes(single_results, meta_results)
        
        print(f"\n{'='*60}")
        print(f"{'Single Worker vs Meta-Reasoner Comparison':^60}")
        print(f"{'='*60}")
        
        print("\nSINGLE WORKER:")
        for key, value in comparison["single_worker"].items():
            print(f"  {key:20s}: {value}")
        
        print("\nMETA-REASONER:")
        for key, value in comparison["meta_reasoner"].items():
            print(f"  {key:20s}: {value}")
        
        print("\nIMPROVEMENTS:")
        improvements = comparison["improvements"]
        print(f"  Accuracy:     {improvements['accuracy_pct']:+.2f}%")
        print(f"  Confidence:   {improvements['confidence_pct']:+.2f}%")
        if improvements['speedup']:
            print(f"  Speedup:      {improvements['speedup']:.2f}x")
        
        print(f"\nRECOMMENDATION:")
        print(f"  {comparison['recommendation']}")
        print(f"{'='*60}\n")
