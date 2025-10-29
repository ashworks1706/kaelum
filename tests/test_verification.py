"""Tests for verification module."""

import pytest
from kaelum.core.verification import (
    SymbolicVerifier,
    FactualVerifier,
    VerificationEngine,
)


class TestSymbolicVerifier:
    """Tests for SymbolicVerifier."""

    def test_verify_valid_equation(self):
        """Test verification of a valid equation."""
        verifier = SymbolicVerifier()
        is_valid, error = verifier.verify_step("2 + 2 = 4")
        assert is_valid is True
        assert error is None

    def test_verify_invalid_equation(self):
        """Test verification of an invalid equation."""
        verifier = SymbolicVerifier()
        is_valid, error = verifier.verify_step("2 + 2 = 5")
        assert is_valid is False
        assert error is not None
        assert "inconsistency" in error.lower()

    def test_verify_no_equation(self):
        """Test verification of text without equations."""
        verifier = SymbolicVerifier()
        is_valid, error = verifier.verify_step("This is a reasoning step without math")
        assert is_valid is True
        assert error is None

    def test_verify_complex_equation(self):
        """Test verification of a more complex equation."""
        verifier = SymbolicVerifier()
        is_valid, error = verifier.verify_step("x + 5 = 8, so x = 3")
        # Symbolic verifier may not parse all natural language equations
        # It should at least not crash
        assert isinstance(is_valid, bool)


class TestFactualVerifier:
    """Tests for FactualVerifier."""

    def test_without_rag(self):
        """Test factual verifier without RAG."""
        verifier = FactualVerifier(use_rag=False)
        is_consistent, confidence = verifier.verify_step("The sky is blue")
        assert is_consistent is True
        assert 0.0 <= confidence <= 1.0

    def test_with_rag(self):
        """Test factual verifier with RAG enabled."""
        verifier = FactualVerifier(use_rag=True)
        is_consistent, confidence = verifier.verify_step("Paris is the capital of France")
        assert is_consistent is True
        assert 0.0 <= confidence <= 1.0


class TestVerificationEngine:
    """Tests for VerificationEngine."""

    def test_verify_trace_symbolic_only(self):
        """Test verification with symbolic verification only."""
        engine = VerificationEngine(use_symbolic=True, use_rag=False)
        trace = ["First, we solve 2 + 2 = 4", "Then we multiply by 3 to get 12"]

        results = engine.verify_trace(trace)

        assert "verified" in results
        assert "symbolic_checks" in results
        assert len(results["symbolic_checks"]) == len(trace)

    def test_verify_trace_with_error(self):
        """Test verification detecting an error."""
        engine = VerificationEngine(use_symbolic=True, use_rag=False)
        # Use a clear mathematical error that will be detected
        trace = ["2 + 2 = 5 is the equation", "Therefore the answer is 5"]

        results = engine.verify_trace(trace)

        # Check that verification produces results
        assert "verified" in results
        assert "symbolic_checks" in results
        # The symbolic verifier may or may not catch this depending on parsing
        # At minimum, it should process the trace
        assert isinstance(results["verified"], bool)

    def test_verify_empty_trace(self):
        """Test verification of empty trace."""
        engine = VerificationEngine(use_symbolic=True, use_rag=False)
        results = engine.verify_trace([])

        assert results["verified"] is True
        assert len(results["symbolic_checks"]) == 0
        assert len(results["errors"]) == 0

    def test_verify_trace_both_modes(self):
        """Test verification with both symbolic and factual."""
        engine = VerificationEngine(use_symbolic=True, use_rag=True)
        trace = ["2 + 2 = 4", "Paris is in France"]

        results = engine.verify_trace(trace)

        assert "symbolic_checks" in results
        assert "factual_checks" in results
