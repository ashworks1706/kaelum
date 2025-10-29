"""Tests for scoring module."""

import pytest
from kaelum.core.scoring import ConfidenceScorer, QualityMetrics


class TestConfidenceScorer:
    """Tests for ConfidenceScorer."""

    def test_compute_confidence_all_valid(self):
        """Test confidence computation with all valid checks."""
        scorer = ConfidenceScorer()

        symbolic_results = {
            "symbolic_checks": [
                {"step": 0, "valid": True},
                {"step": 1, "valid": True},
            ]
        }
        factual_results = {
            "factual_checks": [
                {"step": 0, "confidence": 0.9},
                {"step": 1, "confidence": 0.85},
            ]
        }
        verification_results = {"valid": True, "confidence": 0.9}

        confidence = scorer.compute_confidence(
            symbolic_results, factual_results, verification_results
        )

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7  # Should be high since all checks passed

    def test_compute_confidence_with_errors(self):
        """Test confidence computation with errors."""
        scorer = ConfidenceScorer()

        symbolic_results = {
            "symbolic_checks": [
                {"step": 0, "valid": False},
                {"step": 1, "valid": True},
            ]
        }
        factual_results = {"factual_checks": []}
        verification_results = {"valid": False, "confidence": 0.5}

        confidence = scorer.compute_confidence(
            symbolic_results, factual_results, verification_results
        )

        assert 0.0 <= confidence <= 1.0
        assert confidence < 0.7  # Should be lower due to errors

    def test_compute_step_confidence(self):
        """Test step-level confidence computation."""
        scorer = ConfidenceScorer()

        # Step without errors
        confidence1 = scorer.compute_step_confidence("Valid step", {})
        assert 0.0 <= confidence1 <= 1.0

        # Step with errors
        confidence2 = scorer.compute_step_confidence(
            "Invalid step", {"errors": ["Some error"]}
        )
        assert 0.0 <= confidence2 <= 1.0
        assert confidence2 < confidence1

    def test_aggregate_trace_confidence(self):
        """Test aggregation of step confidences."""
        scorer = ConfidenceScorer()

        # All high confidence
        confidences = [0.9, 0.85, 0.95]
        avg = scorer.aggregate_trace_confidence(confidences)
        assert 0.8 <= avg <= 1.0

        # Mixed confidence
        confidences = [0.9, 0.5, 0.7]
        avg = scorer.aggregate_trace_confidence(confidences)
        assert 0.5 <= avg <= 0.8

        # Empty list
        avg = scorer.aggregate_trace_confidence([])
        assert avg == 0.0


class TestQualityMetrics:
    """Tests for QualityMetrics."""

    def test_initial_metrics(self):
        """Test initial metrics state."""
        metrics = QualityMetrics()
        stats = metrics.get_metrics()

        assert stats["total_requests"] == 0
        assert stats["verified_count"] == 0
        assert stats["failed_count"] == 0
        assert stats["verification_rate"] == 0.0

    def test_record_verified_result(self):
        """Test recording a verified result."""
        metrics = QualityMetrics()
        metrics.record_result(verified=True, confidence=0.9, iterations=2)

        stats = metrics.get_metrics()
        assert stats["total_requests"] == 1
        assert stats["verified_count"] == 1
        assert stats["failed_count"] == 0
        assert stats["verification_rate"] == 1.0
        assert stats["avg_confidence"] == 0.9
        assert stats["avg_iterations"] == 2.0

    def test_record_failed_result(self):
        """Test recording a failed result."""
        metrics = QualityMetrics()
        metrics.record_result(verified=False, confidence=0.5, iterations=1)

        stats = metrics.get_metrics()
        assert stats["total_requests"] == 1
        assert stats["verified_count"] == 0
        assert stats["failed_count"] == 1
        assert stats["verification_rate"] == 0.0

    def test_multiple_results(self):
        """Test recording multiple results."""
        metrics = QualityMetrics()

        metrics.record_result(verified=True, confidence=0.9, iterations=2)
        metrics.record_result(verified=True, confidence=0.8, iterations=1)
        metrics.record_result(verified=False, confidence=0.6, iterations=3)

        stats = metrics.get_metrics()
        assert stats["total_requests"] == 3
        assert stats["verified_count"] == 2
        assert stats["failed_count"] == 1
        assert abs(stats["verification_rate"] - 2/3) < 0.01
        assert abs(stats["avg_confidence"] - 0.767) < 0.01
        assert abs(stats["avg_iterations"] - 2.0) < 0.01

    def test_reset_metrics(self):
        """Test resetting metrics."""
        metrics = QualityMetrics()
        metrics.record_result(verified=True, confidence=0.9, iterations=2)

        metrics.reset()

        stats = metrics.get_metrics()
        assert stats["total_requests"] == 0
        assert stats["verified_count"] == 0
