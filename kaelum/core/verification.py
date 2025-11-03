"""Verification layer for symbolic and factual checking."""

import re
from typing import List, Optional, Tuple
import sympy
from .sympy_engine import SympyEngine


class SymbolicVerifier:
    """Verifies mathematical expressions using SymPy and custom calculus checks.

    Responsibilities:
      - Extract and equivalence-check algebraic equations.
      - Detect derivative/integral transformation steps of the form:
          d/dx(f(x)) = f'(x)
          diff(f(x), x, y) = ∂²f/∂x∂y
          integrate(f(x), x) = F(x) + C   (constant ignored)
      - Gracefully ignore unparsable fragments (never hard-fail verification).
    """

    DERIVATIVE_PATTERN = re.compile(r"(d/d\w+\s*\([^=]+\)|diff\([^=]+\))\s*=\s*[^=]+")
    INTEGRAL_PATTERN = re.compile(r"integrate\([^=]+\)\s*=\s*[^=]+")

    def verify_step(self, step: str) -> Tuple[bool, Optional[str]]:
        """Verify a reasoning step for mathematical correctness.

        Returns (is_valid, error_message) where error_message is None if ok.
        Verification is conservative: only flags definite symbolic mismatches.
        """
        # 1. Derivative checks
        for match in self.DERIVATIVE_PATTERN.findall(step):
            lhs, rhs = match.split('=', 1)
            if not SympyEngine.verify_derivative(lhs.strip(), rhs.strip()):
                return False, f"Incorrect derivative: {match.strip()}"

        # 2. Integral checks
        for match in self.INTEGRAL_PATTERN.findall(step):
            lhs, rhs = match.split('=', 1)
            if not SympyEngine.verify_integral(lhs.strip(), rhs.strip()):
                return False, f"Incorrect integral: {match.strip()}"

        # 3. Equation equivalence checks (algebraic)
        equations = self._extract_equations(step)
        for eq in equations:
            if not self._verify_equation(eq):
                return False, f"Math error: {eq.strip()}"
        return True, None

    def _extract_equations(self, text: str) -> List[str]:
        """Extract mathematical equations from text.

        Avoid re-matching calculus transformation patterns already handled.
        """
        pattern = r"[\w\s\+\-\*/\^\(\)]+\s*(?:==|=)\s*[\w\s\+\-\*/\^\(\)]+"
        equations = re.findall(pattern, text)
        filtered = []
        for eq in equations:
            if self.DERIVATIVE_PATTERN.search(eq) or self.INTEGRAL_PATTERN.search(eq):
                continue
            filtered.append(eq)
        return filtered

    def _verify_equation(self, equation: str) -> bool:
        """Verify algebraic/symbolic equivalence for a single equation."""
        try:
            if '=' in equation:
                left, right = equation.split('=', 1)
                left_expr = sympy.sympify(left.strip())
                right_expr = sympy.sympify(right.strip())
                return sympy.simplify(left_expr - right_expr) == 0
            return True
        except Exception:
            return True  # Silently ignore parsing failures


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
            "factual_passed": 0,
            "calculus_checks": 0,
            "calculus_passed": 0,
        }

        for i, step in enumerate(trace):
            step_passed = True
            
            if self.symbolic_verifier:
                # Count potential calculus transformations before verification
                deriv_matches = self.symbolic_verifier.DERIVATIVE_PATTERN.findall(step)
                integ_matches = self.symbolic_verifier.INTEGRAL_PATTERN.findall(step)
                details["calculus_checks"] += len(deriv_matches) + len(integ_matches)

                is_valid, error = self.symbolic_verifier.verify_step(step)
                details["symbolic_checks"] += 1
                if is_valid:
                    details["symbolic_passed"] += 1
                    # All matched calculus transformations considered passed if whole step valid
                    details["calculus_passed"] += len(deriv_matches) + len(integ_matches)
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
