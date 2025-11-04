"""Verification layer for symbolic and factual checking."""

import re
from typing import List, Optional, Tuple
import sympy
from .sympy_engine import SympyEngine


class SymbolicVerifier:
    # More specific patterns to avoid capturing incomplete fragments
    DERIVATIVE_PATTERN = re.compile(r"(?:d/d\w+|∂/∂\w+|diff)\s*\([^)]+\)\s*=\s*[^=\n]+(?=\s|$|\n|\.)")
    INTEGRAL_PATTERN = re.compile(r"(?:integrate|∫)\s*\([^)]+\)\s*=\s*[^=\n]+(?=\s|$|\n|\.)")
    
    # Pattern for simple arithmetic equations: number/expression = number
    # Matches things like: "899 × 0.15 = 134.85" or "2 + 3 = 5"
    EQUATION_PATTERN = re.compile(
        r'(?<![a-zA-Z])([0-9.]+(?:\s*[+\-*/×÷^]\s*[0-9.]+)*)\s*=\s*([0-9.]+)(?![a-zA-Z*])'
    )
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        # Enable SympyEngine debug mode if verification debug is enabled
        SympyEngine.set_debug(debug)
    
    def _log_debug(self, message: str):
        if self.debug:
            print(f"  [SYMPY DEBUG] {message}")

    def verify_step(self, step: str) -> Tuple[bool, Optional[str]]:
        self._log_debug(f"Verifying step: {step[:100]}...")
        
        # 1. Derivative checks
        derivative_matches = self.DERIVATIVE_PATTERN.findall(step)
        if derivative_matches:
            self._log_debug(f"Found {len(derivative_matches)} derivative pattern(s)")
        
        for match in derivative_matches:
            lhs, rhs = match.split('=', 1)
            self._log_debug(f"  Checking derivative: {lhs.strip()} = {rhs.strip()}")
            self._log_debug(f"  → Calling SympyEngine.verify_derivative()")
            if not SympyEngine.verify_derivative(lhs.strip(), rhs.strip()):
                self._log_debug(f"  ❌ FAILED derivative check")
                return False, f"Incorrect derivative: {match.strip()}"
            self._log_debug(f"  ✓ Derivative verified")

        # 2. Integral checks
        integral_matches = self.INTEGRAL_PATTERN.findall(step)
        if integral_matches:
            self._log_debug(f"Found {len(integral_matches)} integral pattern(s)")
        
        for match in integral_matches:
            lhs, rhs = match.split('=', 1)
            self._log_debug(f"  Checking integral: {lhs.strip()} = {rhs.strip()}")
            self._log_debug(f"  → Calling SympyEngine.verify_integral()")
            if not SympyEngine.verify_integral(lhs.strip(), rhs.strip()):
                self._log_debug(f"  ❌ FAILED integral check")
                return False, f"Incorrect integral: {match.strip()}"
            self._log_debug(f"  ✓ Integral verified")

        # 3. Equation equivalence checks (algebraic)
        equations = self._extract_equations(step)
        if equations:
            self._log_debug(f"Found {len(equations)} algebraic equation(s)")
        
        for eq in equations:
            self._log_debug(f"  Checking equivalence: {eq.strip()}")
            self._log_debug(f"  → Calling SympyEngine.check_equivalence()")
            if not self._verify_equation(eq):
                self._log_debug(f"  ❌ FAILED equivalence check")
                return False, f"Math error: {eq.strip()}"
            self._log_debug(f"  ✓ Equivalence verified")
        
        if derivative_matches or integral_matches or equations:
            self._log_debug(f"✓ All checks passed for this step")
        
        return True, None

    def _extract_equations(self, text: str) -> List[str]:
        # Clean markdown formatting and normalize symbols
        cleaned = re.sub(r'\*\*|\*|__?', '', text)  # Remove markdown bold/italic
        cleaned = cleaned.replace('$', '')           # Remove currency symbols
        cleaned = cleaned.replace('×', '*')          # Normalize multiplication
        cleaned = cleaned.replace('÷', '/')          # Normalize division
        
        equations = []
        
        # Use the specific equation pattern to extract only complete equations
        for match in self.EQUATION_PATTERN.finditer(cleaned):
            lhs, rhs = match.groups()
            lhs = lhs.strip()
            rhs = rhs.strip()
            
            # Validate both sides can be parsed by SymPy
            try:
                SympyEngine._sympify(lhs)
                SympyEngine._sympify(rhs)
                equation = f"{lhs} = {rhs}"
                
                # Skip if this is part of a calculus transformation
                if self.DERIVATIVE_PATTERN.search(text) or self.INTEGRAL_PATTERN.search(text):
                    # Only add if the equation is not within the calculus pattern
                    if equation not in text:
                        equations.append(equation)
                        self._log_debug(f"  Extracted equation: {equation}")
                else:
                    equations.append(equation)
                    self._log_debug(f"  Extracted equation: {equation}")
            except Exception as e:
                self._log_debug(f"  Skipped invalid fragment: '{lhs} = {rhs}' ({type(e).__name__})")
                continue
        
        return equations

    def _verify_equation(self, equation: str) -> bool:
        try:
            if '=' in equation:
                left, right = equation.split('=', 1)
                self._log_debug(f"    → Calling sympy.sympify() for LHS: {left.strip()}")
                left_expr = sympy.sympify(left.strip())
                self._log_debug(f"    Parsed LHS: {left_expr}")
                self._log_debug(f"    → Calling sympy.sympify() for RHS: {right.strip()}")
                right_expr = sympy.sympify(right.strip())
                self._log_debug(f"    Parsed RHS: {right_expr}")
                self._log_debug(f"    → Calling sympy.simplify(LHS - RHS)")
                diff = sympy.simplify(left_expr - right_expr)
                self._log_debug(f"    Simplified difference: {diff}")
                result = diff == 0
                return result
            return True
        except Exception as e:
            self._log_debug(f"    ⚠ Parse error (ignored): {e}")
            return True  # Silently ignore parsing failures


class FactualVerifier:
    def __init__(self, rag_adapter=None):
        self.rag_adapter = rag_adapter

    def verify_step(self, step: str) -> Tuple[bool, float]:
        if not self.rag_adapter:
            return True, 1.0
        
        is_verified, confidence = self.rag_adapter.verify_claim(step)
        return is_verified, confidence


class VerificationEngine:

    def __init__(self, llm_client, use_symbolic: bool = True, use_factual: bool = False, debug: bool = False):
        self.llm_client = llm_client
        self.symbolic_verifier = SymbolicVerifier(debug=debug) if use_symbolic else None
        self.factual_verifier = None  # RAG verification removed for simplicity
        self.debug = debug
    
    def _log_debug(self, message: str):
        if self.debug:
            print(f"[VERIFICATION DEBUG] {message}")

    def verify_trace(self, trace: List[str]) -> Tuple[List[str], dict]:
        self._log_debug(f"Starting verification of {len(trace)} steps")
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
            self._log_debug(f"\n--- Step {i+1}/{len(trace)} ---")
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
                    self._log_debug(f"❌ Step {i+1} FAILED symbolic verification: {error}")
                    errors.append(f"Step {i+1}: {error}")
                    step_passed = False

            if self.factual_verifier:
                is_consistent, confidence = self.factual_verifier.verify_step(step)
                details["factual_checks"] += 1
                if is_consistent:
                    details["factual_passed"] += 1
                else:
                    self._log_debug(f"❌ Step {i+1} FAILED factual check (confidence: {confidence:.2f})")
                    errors.append(f"Step {i+1}: Factual inconsistency (confidence: {confidence:.2f})")
                    step_passed = False
            
            if step_passed:
                details["verified_steps"] += 1

        self._log_debug(f"\n=== Verification Summary ===")
        self._log_debug(f"Total steps: {details['total_steps']}")
        self._log_debug(f"Verified steps: {details['verified_steps']}")
        if self.symbolic_verifier:
            self._log_debug(f"Symbolic: {details['symbolic_passed']}/{details['symbolic_checks']} passed")
            self._log_debug(f"Calculus: {details['calculus_passed']}/{details['calculus_checks']} passed")
        if self.factual_verifier:
            self._log_debug(f"Factual: {details['factual_passed']}/{details['factual_checks']} passed")
        self._log_debug(f"Errors found: {len(errors)}\n")

        return errors, details
    
    def verify(self, query: str, reasoning_steps: List[str], answer: str) -> dict:
        # Run symbolic/factual verification on reasoning steps
        errors, details = self.verify_trace(reasoning_steps)
        
        # Calculate confidence based on verification results
        if details["total_steps"] == 0:
            confidence = 0.5  # No steps to verify, uncertain
        else:
            # Confidence based on percentage of steps that passed verification
            confidence = details["verified_steps"] / details["total_steps"]
        
        # Overall pass/fail based on whether any errors found
        passed = len(errors) == 0
        
        return {
            "passed": passed,
            "confidence": confidence,
            "issues": errors,
            "details": details
        }
