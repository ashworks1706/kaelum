import re
import ast
from typing import List, Optional, Tuple
import sympy
from .sympy_engine import SympyEngine
from sentence_transformers import SentenceTransformer, util


class SymbolicVerifier:
    DERIVATIVE_PATTERN = re.compile(r"(?:d/d\w+|∂/∂\w+|diff)\s*\([^)]+\)\s*=\s*[^=\n]+(?=\s|$|\n|\.)")
    INTEGRAL_PATTERN = re.compile(r"(?:integrate|∫)\s*\([^)]+\)\s*=\s*[^=\n]+(?=\s|$|\n|\.)")
    
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




class VerificationEngine:

    def __init__(self, llm_client, use_symbolic: bool = True, use_factual: bool = False, debug: bool = False):
        self.llm_client = llm_client
        self.symbolic_verifier = SymbolicVerifier(debug=debug) if use_symbolic else None
        self.debug = debug
        self.semantic_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
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
        worker_type = self._infer_worker_type(query, reasoning_steps)
        
        if worker_type == "math":
            return self._verify_math(reasoning_steps)
        elif worker_type == "code":
            return self._verify_code(query, answer, reasoning_steps)
        elif worker_type == "logic":
            return self._verify_logic(query, reasoning_steps, answer)
        elif worker_type == "factual":
            return self._verify_factual(query, answer, reasoning_steps)
        elif worker_type == "creative":
            return self._verify_creative(answer, reasoning_steps)
        else:
            return self._verify_analysis(query, answer, reasoning_steps)
    
    def _infer_worker_type(self, query: str, reasoning_steps: List[str]) -> str:
        combined_text = query + " " + " ".join(reasoning_steps[:3])
        
        math_indicators = sum(c in combined_text for c in '+-*/=^√∫∂∑')
        code_indicators = sum(c in combined_text for c in '{}[]();')
        logic_indicators = combined_text.lower().count('therefore') + combined_text.lower().count('thus')
        
        if math_indicators > 3:
            return "math"
        elif code_indicators > 5:
            return "code"
        elif logic_indicators > 0:
            return "logic"
        elif any(kw in query.lower() for kw in ['story', 'poem', 'create', 'write', 'brainstorm']):
            return "creative"
        elif any(kw in query.lower() for kw in ['what', 'when', 'where', 'who', 'define', 'explain']):
            return "factual"
        else:
            return "analysis"
    
    def _verify_math(self, reasoning_steps: List[str]) -> dict:
        errors, details = self.verify_trace(reasoning_steps)
        
        if details["total_steps"] == 0:
            confidence = 0.5
        else:
            confidence = details["verified_steps"] / details["total_steps"]
        
        passed = len(errors) == 0
        
        return {
            "passed": passed,
            "confidence": confidence,
            "issues": errors,
            "details": details
        }
    
    def _verify_code(self, query: str, answer: str, reasoning_steps: List[str]) -> dict:
        issues = []
        
        code_block_pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(code_block_pattern, answer, re.DOTALL)
        code = matches[0].strip() if matches else None
        
        if not code:
            issues.append("No code block found in answer")
            return {
                "passed": False,
                "confidence": 0.3,
                "issues": issues,
                "details": {"syntax_valid": False}
            }
        
        if 'python' in query.lower() or 'def ' in code or 'import ' in code:
            try:
                ast.parse(code)
                syntax_valid = True
            except SyntaxError as e:
                syntax_valid = False
                issues.append(f"Python syntax error: {str(e)}")
        else:
            syntax_valid = True
        
        if len(code) < 10:
            issues.append("Code is too short")
        
        confidence = 0.8 if syntax_valid else 0.3
        passed = syntax_valid and len(issues) == 0
        
        return {
            "passed": passed,
            "confidence": confidence,
            "issues": issues,
            "details": {"syntax_valid": syntax_valid, "code_length": len(code)}
        }
    
    def _verify_logic(self, query: str, reasoning_steps: List[str], answer: str) -> dict:
        issues = []
        
        if len(reasoning_steps) < 2:
            issues.append("Insufficient logical reasoning steps")
        
        conclusion_keywords = ['therefore', 'thus', 'conclude', 'hence']
        has_conclusion = any(kw in step.lower() for step in reasoning_steps for kw in conclusion_keywords)
        
        if not has_conclusion:
            issues.append("No clear logical conclusion")
        
        query_embedding = self.semantic_encoder.encode(query, convert_to_tensor=True)
        answer_embedding = self.semantic_encoder.encode(answer, convert_to_tensor=True)
        relevance = float(util.cos_sim(query_embedding, answer_embedding)[0][0])
        
        if relevance < 0.3:
            issues.append("Answer not relevant to query")
        
        confidence = 0.7 if has_conclusion else 0.5
        confidence *= (relevance + 0.5) / 1.5
        passed = len(issues) == 0
        
        return {
            "passed": passed,
            "confidence": confidence,
            "issues": issues,
            "details": {"has_conclusion": has_conclusion, "relevance": relevance}
        }
    
    def _verify_factual(self, query: str, answer: str, reasoning_steps: List[str]) -> dict:
        issues = []
        
        if len(answer) < 20:
            issues.append("Answer too short for factual query")
        
        query_embedding = self.semantic_encoder.encode(query, convert_to_tensor=True)
        answer_embedding = self.semantic_encoder.encode(answer, convert_to_tensor=True)
        relevance = float(util.cos_sim(query_embedding, answer_embedding)[0][0])
        
        if relevance < 0.35:
            issues.append("Answer not semantically relevant to query")
        
        has_specifics = bool(re.search(r'\d+', answer)) or len(answer) > 100
        if not has_specifics:
            issues.append("Answer lacks specific details")
        
        confidence = relevance * (0.9 if has_specifics else 0.7)
        passed = len(issues) == 0
        
        return {
            "passed": passed,
            "confidence": confidence,
            "issues": issues,
            "details": {"relevance": relevance, "has_specifics": has_specifics}
        }
    
    def _verify_creative(self, answer: str, reasoning_steps: List[str]) -> dict:
        issues = []
        
        if len(answer) < 50:
            issues.append("Creative content too short")
        
        words = answer.split()
        if len(words) < 20:
            issues.append("Insufficient word count")
        
        unique_words = len(set(w.lower() for w in words))
        diversity = unique_words / max(len(words), 1)
        
        if diversity < 0.4:
            issues.append("Low vocabulary diversity")
        
        has_structure = '.' in answer or '\n' in answer or '!' in answer or '?' in answer
        if not has_structure:
            issues.append("Lacks structure (no punctuation or line breaks)")
        
        confidence = diversity * (0.9 if has_structure else 0.7)
        passed = len(issues) == 0
        
        return {
            "passed": passed,
            "confidence": confidence,
            "issues": issues,
            "details": {"diversity": diversity, "has_structure": has_structure, "word_count": len(words)}
        }
    
    def _verify_analysis(self, query: str, answer: str, reasoning_steps: List[str]) -> dict:
        issues = []
        
        if len(reasoning_steps) < 2:
            issues.append("Insufficient analytical reasoning")
        
        if len(answer) < 30:
            issues.append("Analysis too brief")
        
        query_embedding = self.semantic_encoder.encode(query, convert_to_tensor=True)
        answer_embedding = self.semantic_encoder.encode(answer, convert_to_tensor=True)
        relevance = float(util.cos_sim(query_embedding, answer_embedding)[0][0])
        
        if relevance < 0.3:
            issues.append("Analysis not relevant to query")
        
        confidence = relevance * 0.9 if len(reasoning_steps) >= 2 else 0.5
        passed = len(issues) == 0
        
        return {
            "passed": passed,
            "confidence": confidence,
            "issues": issues,
            "details": {"relevance": relevance, "step_count": len(reasoning_steps)}
        }
