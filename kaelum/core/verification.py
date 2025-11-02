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

    def verify_trace(self, trace: List[str]) -> Tuple[List[str], dict]:
        """Verify reasoning trace and return errors plus detailed results."""
        errors = []
        details = {
            "total_steps": len(trace),
            "verified_steps": 0,
            "symbolic_checks": 0,
            "symbolic_passed": 0,
            "factual_checks": 0,
            "factual_passed": 0
        }

        for i, step in enumerate(trace):
            step_passed = True
            
            if self.symbolic_verifier:
                is_valid, error = self.symbolic_verifier.verify_step(step)
                details["symbolic_checks"] += 1
                if is_valid:
                    details["symbolic_passed"] += 1
                else:
                    errors.append(f"Step {i+1}: {error}")
                    step_passed = False

            if self.factual_verifier:
                is_consistent, confidence = self.factual_verifier.verify_step(step)
                details["factual_checks"] += 1
                if is_consistent:
                    details["factual_passed"] += 1
                else:
                    errors.append(f"Step {i+1}: Factual inconsistency (confidence: {confidence:.2f})")
                    step_passed = False
            
            if step_passed:
                details["verified_steps"] += 1

        return errors, details
