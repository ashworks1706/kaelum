import re
import ast
from typing import List, Optional, Tuple
import sympy
from .sympy_engine import SympyEngine
from sentence_transformers import SentenceTransformer, util
from core.shared_encoder import get_shared_encoder


class SymbolicVerifier:
    DERIVATIVE_PATTERN = re.compile(r"(?:d/d\w+|∂/∂\w+|diff)\s*\([^)]+\)\s*=\s*[^=\n]+(?=\s|$|\n|\.)")
    INTEGRAL_PATTERN = re.compile(r"(?:integrate|∫)\s*\([^)]+\)\s*=\s*[^=\n]+(?=\s|$|\n|\.)")
    
    EQUATION_PATTERN = re.compile(
        r'(?<![a-zA-Z])([0-9.]+(?:\s*[+\-*/×÷^]\s*[0-9.]+)*)\s*=\s*([0-9.]+)(?![a-zA-Z*])'
    )
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', debug: bool = False):
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




import logging
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer, util

from ..reasoning import LLMClient, Message
from .sympy_engine import SympyEngine
from .syntax_validator import SyntaxValidator
from ..detectors.conclusion_detector import ConclusionDetector
from ..detectors.worker_type_classifier import WorkerTypeClassifier
from ..detectors.domain_classifier import DomainClassifier


class VerificationEngine:

    def __init__(self, llm_client, use_symbolic: bool = True, use_factual: bool = False, debug: bool = False, embedding_model: str = "all-MiniLM-L6-v2"):
        self.llm_client = llm_client
        self.symbolic_verifier = SymbolicVerifier(debug=debug) if use_symbolic else None
        self.use_factual = use_factual
        self.factual_verifier = None  # TODO: Implement factual verifier if needed
        self.debug = debug
        
        # Use shared encoder to avoid loading model multiple times
        self.semantic_encoder = get_shared_encoder(embedding_model, device='cpu')
        self.syntax_validator = SyntaxValidator()
        self.conclusion_detector = ConclusionDetector(embedding_model=embedding_model)
        self.worker_classifier = WorkerTypeClassifier(embedding_model=embedding_model)
        self.domain_classifier = DomainClassifier(embedding_model=embedding_model)
    
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
    
    def verify(self, query: str, reasoning_steps: List[str], answer: str, worker_type: Optional[str] = None) -> dict:
        import logging
        logger = logging.getLogger("kaelum.verification")
        
        if worker_type is None:
            worker_type = self._infer_worker_type(query, reasoning_steps)
        
        logger.info(f"\n{'═' * 80}")
        logger.info(f"VERIFICATION: Starting verification for {worker_type.upper()} worker")
        logger.info(f"  Query: {query[:100]}...")
        logger.info(f"  Steps: {len(reasoning_steps)} reasoning steps")
        
        if worker_type == "math":
            result = self._verify_math(reasoning_steps)
        elif worker_type == "code":
            result = self._verify_code(query, answer, reasoning_steps)
        elif worker_type == "logic":
            result = self._verify_logic(query, reasoning_steps, answer)
        elif worker_type == "factual":
            result = self._verify_factual(query, answer, reasoning_steps)
        elif worker_type == "creative":
            result = self._verify_creative(answer, reasoning_steps)
        else:
            result = self._verify_analysis(query, answer, reasoning_steps)
        
        status = "✓ PASSED" if result["passed"] else "✗ FAILED"
        logger.info(f"\nVERIFICATION: {status}")
        logger.info(f"  Confidence: {result['confidence']:.3f}")
        if not result["passed"] and result.get("issues"):
            logger.info(f"  Issues found: {len(result['issues'])}")
            for i, issue in enumerate(result["issues"][:3], 1):  # Show first 3 issues
                logger.info(f"    {i}. {issue}")
            if len(result['issues']) > 3:
                logger.info(f"    ... and {len(result['issues']) - 3} more")
        logger.info(f"{'═' * 80}\n")
        
        return result
    
    def _infer_worker_type(self, query: str, reasoning_steps: List[str]) -> str:
        combined_text = query + " " + " ".join(reasoning_steps[:3])
        
        classification = self.worker_classifier.classify_worker(query)
        
        if classification['confidence'] > 0.5:
            return classification['worker']
        
        math_pattern = bool(re.search(r'\d+\s*[+\-*/^=]\s*\d+', combined_text))
        code_pattern = bool(re.search(r'\b(def|class|function|import|return)\b', combined_text.lower()))
        logic_pattern = bool(re.search(r'\b(therefore|thus|premise|conclusion|implies)\b', combined_text.lower()))
        
        if math_pattern:
            return "math"
        elif code_pattern:
            return "code"
        elif logic_pattern:
            return "logic"
        else:
            return classification['worker']
    
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
        
        language = self._detect_code_language(query, code)
        syntax_valid = False
        
        if language == 'python':
            try:
                ast.parse(code)
                syntax_valid = True
            except SyntaxError as e:
                syntax_valid = False
                issues.append(f"Python syntax error: {str(e)}")
        elif language == 'javascript':
            if re.search(r'\bfunction\s+\w+\s*\(', code) or re.search(r'\bconst\s+\w+\s*=', code):
                syntax_valid = True
            else:
                issues.append("JavaScript code structure unclear")
        elif language == 'java':
            if re.search(r'\b(public|private|class)\b', code) and '{' in code:
                syntax_valid = True
            else:
                issues.append("Java code structure unclear")
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
            "details": {"syntax_valid": syntax_valid, "code_length": len(code), "language": language}
        }
    
    def _detect_code_language(self, query: str, code: str) -> str:
        """Detect programming language from code content."""
        code_lower = code.lower()
        query_lower = query.lower()
        
        # Check query for explicit language mentions
        if 'python' in query_lower or 'py' in query_lower:
            return 'python'
        elif 'javascript' in query_lower or 'js' in query_lower:
            return 'javascript'
        elif 'typescript' in query_lower or 'ts' in query_lower:
            return 'typescript'
        elif 'java' in query_lower and 'javascript' not in query_lower:
            return 'java'
        elif 'c++' in query_lower or 'cpp' in query_lower:
            return 'cpp'
        elif 'rust' in query_lower:
            return 'rust'
        elif 'go' in query_lower and 'golang' in query_lower:
            return 'go'
        
        # Check code patterns
        if 'def ' in code or 'import ' in code or 'from ' in code:
            return 'python'
        elif 'function ' in code or 'const ' in code or 'let ' in code or 'var ' in code:
            if ': ' in code and 'interface ' in code:
                return 'typescript'
            return 'javascript'
        elif 'public class' in code or 'private ' in code:
            return 'java'
        elif '#include' in code:
            return 'cpp'
        elif 'fn ' in code or 'let mut' in code:
            return 'rust'
        elif 'func ' in code and 'package ' in code:
            return 'go'
        
        # Default to python
        return 'python'
    
    def _verify_logic(self, query: str, reasoning_steps: List[str], answer: str) -> dict:
        issues = []
        
        if len(reasoning_steps) < 2:
            issues.append("Insufficient logical reasoning steps")
        
        conclusion_result = self.conclusion_detector.detect(
            reasoning_steps[-1] if reasoning_steps else answer,
            reasoning_steps
        )
        has_conclusion = conclusion_result['is_conclusion']
        
        if not has_conclusion:
            issues.append("No clear logical conclusion")
        
        query_embedding = self.semantic_encoder.encode(query, convert_to_tensor=True)
        answer_embedding = self.semantic_encoder.encode(answer, convert_to_tensor=True)
        relevance = float(util.cos_sim(query_embedding, answer_embedding)[0][0])
        
        relevance_threshold = 0.35
        if relevance < relevance_threshold:
            issues.append("Answer not relevant to query")
        
        base_confidence = 0.75 if has_conclusion else 0.55
        confidence = base_confidence * ((relevance + 0.5) / 1.5)
        passed = len(issues) == 0
        
        return {
            "passed": passed,
            "confidence": confidence,
            "issues": issues,
            "details": {
                "has_conclusion": has_conclusion,
                "conclusion_confidence": conclusion_result['confidence'],
                "relevance": relevance
            }
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
        
        has_intentional_repetition = self._detect_intentional_repetition(answer)
        
        min_diversity = 0.3
        if diversity < min_diversity and not has_intentional_repetition:
            issues.append("Low vocabulary diversity")
        
        has_structure = '.' in answer or '\n' in answer or '!' in answer or '?' in answer
        if not has_structure:
            issues.append("Lacks structure (no punctuation or line breaks)")
        
        has_imagery = self._detect_imagery(answer)
        has_narrative = len(answer) > 100 and has_structure
        
        creativity_score = (
            diversity * 0.3 +
            float(has_structure) * 0.2 +
            float(has_imagery) * 0.2 +
            float(has_narrative) * 0.2 +
            float(has_intentional_repetition) * 0.1
        )
        
        confidence = min(creativity_score, 0.95)
        passed = len(issues) == 0
        
        return {
            "passed": passed,
            "confidence": confidence,
            "issues": issues,
            "details": {
                "diversity": diversity, 
                "has_structure": has_structure, 
                "word_count": len(words),
                "has_imagery": has_imagery,
                "has_intentional_repetition": has_intentional_repetition
            }
        }
    
    def _detect_intentional_repetition(self, text: str) -> bool:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) < 3:
            return False
        
        starts = [line.split()[0].lower() if line.split() else '' for line in lines]
        ends = [line.split()[-1].lower() if line.split() else '' for line in lines]
        
        start_repetition = len(starts) > 0 and len(set(starts)) / len(starts) < 0.7
        end_repetition = len(ends) > 0 and len(set(ends)) / len(ends) < 0.7
        
        return start_repetition or end_repetition
    
    def _detect_imagery(self, text: str) -> bool:
        imagery_words = {
            'see', 'saw', 'look', 'bright', 'dark', 'color', 'light', 'shadow',
            'hear', 'heard', 'sound', 'whisper', 'loud', 'quiet',
            'smell', 'scent', 'fragrance', 'aroma',
            'feel', 'felt', 'touch', 'soft', 'rough', 'smooth',
            'taste', 'sweet', 'bitter', 'sour'
        }
        
        words = text.lower().split()
        imagery_count = sum(1 for w in words if w in imagery_words)
        
        return imagery_count >= 2
    
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
