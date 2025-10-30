"""Verification layer for symbolic and factual checking."""

import re
from typing import Dict, List, Optional, Tuple

import sympy


class SymbolicVerifier:
    """Verifies mathematical and logical expressions using SymPy."""

    def __init__(self):
        """Initialize symbolic verifier."""
        self.sympy_namespace = {
            "symbols": sympy.symbols,
            "sympify": sympy.sympify,
            "solve": sympy.solve,
            "simplify": sympy.simplify,
            "expand": sympy.expand,
            "factor": sympy.factor,
        }

    def verify_step(self, step: str) -> Tuple[bool, Optional[str]]:
        """
        Verify a single reasoning step for mathematical correctness.

        Returns:
            (is_valid, error_message)
        """
        # Extract mathematical expressions
        equations = self._extract_equations(step)

        if not equations:
            # No mathematical content to verify
            return True, None

        for eq in equations:
            # Parse and verify - fail loudly if there's an issue
            result = self._verify_equation(eq)
            if not result:
                return False, f"Mathematical inconsistency detected in: {eq}"

        return True, None

    def _extract_equations(self, text: str) -> List[str]:
        """Extract mathematical equations from text."""
        # Look for patterns like: x = ..., 2 + 2 = 4, etc.
        patterns = [
            r"[\w\s\+\-\*/\(\)]+\s*=\s*[\w\s\+\-\*/\(\)]+",
            r"\$[^\$]+\$",  # LaTeX style
        ]

        equations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            equations.extend(matches)

        return equations

    def _verify_equation(self, equation: str) -> bool:
        """Verify a single equation. Fails loudly on parse errors."""
        # Remove $ signs (LaTeX)
        equation = equation.replace("$", "").strip()

        # Parse both sides - no fallback
        if "=" in equation:
            left, right = equation.split("=", 1)
            left_expr = sympy.sympify(left.strip())
            right_expr = sympy.sympify(right.strip())

            # Check if they're equivalent
            diff = sympy.simplify(left_expr - right_expr)
            return diff == 0

        return True


class FactualVerifier:
    """Verifies factual claims using simple heuristics, patterns, or RAG."""

    def __init__(self, use_factual_check: bool = False, rag_adapter=None):
        """
        Initialize factual verifier.
        
        Args:
            use_factual_check: Enable factual verification
            rag_adapter: Optional RAG adapter (ChromaAdapter, QdrantAdapter, etc.)
        """
        self.use_factual_check = use_factual_check
        self.rag_adapter = rag_adapter
        
        # Simple patterns for common factual claims
        self.claim_patterns = {
            'numerical': r'\d+',
            'date': r'\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
            'location': r'\b(?:in|at|near|from)\s+[A-Z][a-z]+',
        }

    def verify_step(self, step: str, context: Optional[List[str]] = None) -> Tuple[bool, float]:
        """
        Verify a factual claim using RAG adapter. No fallbacks.

        Returns:
            (is_consistent, confidence)
        """
        if not self.use_factual_check:
            # Skip verification if not enabled
            return True, 1.0

        # RAG adapter is REQUIRED if factual checking is enabled
        if not self.rag_adapter:
            raise ValueError(
                "Factual verification is enabled but no RAG adapter provided. "
                "Pass a RAG adapter (ChromaAdapter, QdrantAdapter, etc.) or disable factual verification."
            )

        # Use RAG adapter - fail loudly if it fails
        is_verified, confidence = self.rag_adapter.verify_claim(step, context)
        return is_verified, confidence


class VerificationEngine:
    """Combines symbolic and factual verification."""

    def __init__(self, use_symbolic: bool = True, use_factual_check: bool = False, rag_adapter=None):
        """
        Initialize verification engine.
        
        Args:
            use_symbolic: Enable symbolic verification
            use_factual_check: Enable factual verification
            rag_adapter: Optional RAG adapter for factual verification
        """
        self.symbolic_verifier = SymbolicVerifier() if use_symbolic else None
        self.factual_verifier = FactualVerifier(use_factual_check=use_factual_check, rag_adapter=rag_adapter)

    def verify_trace(self, trace: List[str]) -> Dict[str, any]:
        """
        Verify a complete reasoning trace.

        Returns:
            Dictionary with verification results
        """
        results = {
            "verified": True,
            "symbolic_checks": [],
            "factual_checks": [],
            "errors": [],
        }

        for i, step in enumerate(trace):
            # Symbolic verification
            if self.symbolic_verifier:
                is_valid, error = self.symbolic_verifier.verify_step(step)
                results["symbolic_checks"].append(
                    {"step": i, "valid": is_valid, "error": error}
                )
                if not is_valid:
                    results["verified"] = False
                    results["errors"].append(f"Step {i+1}: {error}")

            # Factual verification
            if self.factual_verifier:
                is_consistent, confidence = self.factual_verifier.verify_step(step)
                results["factual_checks"].append(
                    {"step": i, "consistent": is_consistent, "confidence": confidence}
                )
                if not is_consistent:
                    results["verified"] = False
                    results["errors"].append(f"Step {i+1}: Factual inconsistency detected")

        return results
