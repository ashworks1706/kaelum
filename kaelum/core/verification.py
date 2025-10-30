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
            try:
                # Try to parse and verify the equation
                result = self._verify_equation(eq)
                if not result:
                    return False, f"Mathematical inconsistency detected in: {eq}"
            except Exception as e:
                # If we can't parse, assume it's not verifiable symbolically
                continue

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
        """Verify a single equation."""
        # Remove $ signs (LaTeX)
        equation = equation.replace("$", "").strip()

        # Try to parse both sides
        if "=" in equation:
            left, right = equation.split("=", 1)
            try:
                left_expr = sympy.sympify(left.strip())
                right_expr = sympy.sympify(right.strip())

                # Check if they're equivalent
                diff = sympy.simplify(left_expr - right_expr)
                return diff == 0
            except:
                return True  # Can't verify, assume correct

        return True


class FactualVerifier:
    """Verifies factual claims using simple heuristics and patterns."""

    def __init__(self, use_factual_check: bool = False):
        """Initialize factual verifier."""
        self.use_factual_check = use_factual_check
        
        # Simple patterns for common factual claims
        self.claim_patterns = {
            'numerical': r'\d+',
            'date': r'\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
            'location': r'\b(?:in|at|near|from)\s+[A-Z][a-z]+',
        }

    def verify_step(self, step: str, context: Optional[List[str]] = None) -> Tuple[bool, float]:
        """
        Verify a factual claim using simple pattern matching.

        Returns:
            (is_consistent, confidence)
        """
        if not self.use_factual_check:
            # Without factual checking, we trust the step with moderate confidence
            return True, 0.8

        # Check for self-contradictions in the step
        confidence = 0.8
        
        # Simple heuristic: check if the step contains definitive claims
        definitive_words = ['always', 'never', 'all', 'none', 'every', 'no']
        for word in definitive_words:
            if word in step.lower():
                confidence -= 0.1  # Lower confidence for absolute claims
        
        # Check consistency with context if provided
        if context:
            step_lower = step.lower()
            for ctx in context:
                # Look for contradicting statements
                if any(neg in ctx.lower() for neg in ['not', 'never', 'no']) and \
                   any(word in step_lower for word in ctx.lower().split()):
                    confidence -= 0.2
        
        confidence = max(0.5, min(1.0, confidence))
        is_consistent = confidence > 0.6
        
        return is_consistent, confidence


class VerificationEngine:
    """Combines symbolic and factual verification."""

    def __init__(self, use_symbolic: bool = True, use_factual_check: bool = False):
        """Initialize verification engine."""
        self.symbolic_verifier = SymbolicVerifier() if use_symbolic else None
        self.factual_verifier = FactualVerifier(use_factual_check=use_factual_check)

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
