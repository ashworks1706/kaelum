"""Confidence scoring engine for reasoning quality assessment."""

from typing import Dict, List

import numpy as np


class ConfidenceScorer:
    """Quantifies reliability of reasoning traces."""

    def __init__(self):
        """Initialize confidence scorer."""
        self.weights = {
            "symbolic_verification": 0.3,
            "factual_verification": 0.2,
            "verifier_confidence": 0.3,
            "logical_consistency": 0.2,
        }

    def compute_confidence(
        self,
        symbolic_results: Dict,
        factual_results: Dict,
        verification_results: Dict,
    ) -> float:
        """
        Compute overall confidence score for a reasoning trace.

        Args:
            symbolic_results: Results from symbolic verification
            factual_results: Results from factual verification
            verification_results: Results from LLM verifier

        Returns:
            Confidence score between 0 and 1
        """
        scores = []

        # Symbolic verification score
        if symbolic_results.get("symbolic_checks"):
            checks = symbolic_results["symbolic_checks"]
            valid_count = sum(1 for c in checks if c["valid"])
            symbolic_score = valid_count / len(checks) if checks else 1.0
            scores.append(("symbolic_verification", symbolic_score))

        # Factual verification score
        if factual_results.get("factual_checks"):
            checks = factual_results["factual_checks"]
            avg_confidence = np.mean([c["confidence"] for c in checks])
            scores.append(("factual_verification", avg_confidence))

        # Verifier confidence
        verifier_conf = verification_results.get("confidence", 0.8)
        scores.append(("verifier_confidence", verifier_conf))

        # Logical consistency (from verification validity)
        consistency_score = 1.0 if verification_results.get("valid", True) else 0.5
        scores.append(("logical_consistency", consistency_score))

        # Weighted average
        total_confidence = sum(
            score * self.weights.get(name, 0.25) for name, score in scores
        )

        # Normalize to [0, 1]
        return min(max(total_confidence, 0.0), 1.0)

    def compute_step_confidence(self, step: str, verification_data: Dict) -> float:
        """Compute confidence for a single reasoning step."""
        # Simple heuristic based on step length and verification
        base_confidence = 0.7

        # Adjust based on verification data
        if verification_data.get("errors"):
            base_confidence *= 0.5

        return base_confidence

    def aggregate_trace_confidence(self, step_confidences: List[float]) -> float:
        """Aggregate step-level confidences into trace-level confidence."""
        if not step_confidences:
            return 0.0

        # Use harmonic mean (more conservative than arithmetic mean)
        harmonic_mean = len(step_confidences) / sum(1 / (c + 0.01) for c in step_confidences)

        return min(max(harmonic_mean, 0.0), 1.0)


class QualityMetrics:
    """Tracks reasoning quality metrics for analysis."""

    def __init__(self):
        """Initialize quality metrics tracker."""
        self.metrics = {
            "total_requests": 0,
            "verified_count": 0,
            "failed_count": 0,
            "avg_confidence": 0.0,
            "avg_iterations": 0.0,
            "confidences": [],
            "iterations": [],
        }

    def record_result(
        self, verified: bool, confidence: float, iterations: int
    ) -> None:
        """Record a reasoning result."""
        self.metrics["total_requests"] += 1

        if verified:
            self.metrics["verified_count"] += 1
        else:
            self.metrics["failed_count"] += 1

        self.metrics["confidences"].append(confidence)
        self.metrics["iterations"].append(iterations)

        # Update averages
        self.metrics["avg_confidence"] = np.mean(self.metrics["confidences"])
        self.metrics["avg_iterations"] = np.mean(self.metrics["iterations"])

    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return {
            "total_requests": self.metrics["total_requests"],
            "verified_count": self.metrics["verified_count"],
            "failed_count": self.metrics["failed_count"],
            "verification_rate": (
                self.metrics["verified_count"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0
                else 0.0
            ),
            "avg_confidence": float(self.metrics["avg_confidence"]),
            "avg_iterations": float(self.metrics["avg_iterations"]),
        }

    def reset(self) -> None:
        """Reset metrics."""
        self.__init__()
