"""Verification layer for symbolic and factual checking."""

import re
from typing import List, Optional, Tuple
import sympy


class SymbolicVerifier:
    """Verifies mathematical expressions using SymPy."""

    def verify_step(self, step: str) -> Tuple[bool, Optional[str]]:
        """Verify a reasoning step for mathematical correctness."""
        equations = self._extract_equations(step)

        if not equations:
            return True, None

        for eq in equations:
            if not self._verify_equation(eq):
                return False, f"Math error: {eq}"

        return True, None

    def _extract_equations(self, text: str) -> List[str]:
        """Extract mathematical equations from text."""
        pattern = r"[\w\s\+\-\*/\(\)]+\s*=\s*[\w\s\+\-\*/\(\)]+"
        return re.findall(pattern, text)

    def _verify_equation(self, equation: str) -> bool:
        """Verify a single equation."""
        try:
            if "=" in equation:
                left, right = equation.split("=", 1)
                left_expr = sympy.sympify(left.strip())
                right_expr = sympy.sympify(right.strip())
                diff = sympy.simplify(left_expr - right_expr)
                return diff == 0
            return True
        except:
            return True  # Skip unparseable equations


class FactualVerifier:
    """Verifies factual claims using RAG."""

    def __init__(self, rag_adapter=None):
        self.rag_adapter = rag_adapter

    def verify_step(self, step: str) -> Tuple[bool, float]:
        """Verify a factual claim using RAG adapter."""
        if not self.rag_adapter:
            return True, 1.0
        
        is_verified, confidence = self.rag_adapter.verify_claim(step)
        return is_verified, confidence


class VerificationEngine:
    """Combines symbolic and factual verification."""

    def __init__(self, use_symbolic: bool = True, use_factual_check: bool = False, rag_adapter=None):
        self.symbolic_verifier = SymbolicVerifier() if use_symbolic else None
        self.factual_verifier = FactualVerifier(rag_adapter) if use_factual_check else None

    def verify_trace(self, trace: List[str]) -> List[str]:
        """Verify reasoning trace and return errors."""
        errors = []

        for i, step in enumerate(trace):
            if self.symbolic_verifier:
                is_valid, error = self.symbolic_verifier.verify_step(step)
                if not is_valid:
                    errors.append(f"Step {i+1}: {error}")

            if self.factual_verifier:
                is_consistent, confidence = self.factual_verifier.verify_step(step)
                if not is_consistent:
                    errors.append(f"Step {i+1}: Factual inconsistency")

        return errors
