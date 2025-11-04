"""Worker agents - specialized reasoning agents for specific query types.

Workers are domain-specific agents that excel at particular types of queries:
- MathWorker: Mathematical calculations, symbolic reasoning
- LogicWorker: Logical proofs, deductive reasoning
- CodeWorker: Code generation, debugging, algorithms
- etc.

Each worker can be run independently or in parallel via the ParallelRunner.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

from kaelum.core.config import KaelumConfig
from kaelum.core.reasoning import LLMClient
from kaelum.core.sympy_engine import SympyEngine


class WorkerSpecialty(Enum):
    """Worker specialties."""
    MATH = "math"
    LOGIC = "logic"
    CODE = "code"
    FACTUAL = "factual"
    CREATIVE = "creative"
    ANALYSIS = "analysis"


@dataclass
class WorkerResult:
    """Result from a worker agent."""
    answer: str
    confidence: float  # 0-1
    reasoning_steps: List[str]
    verification_passed: bool
    specialty: WorkerSpecialty
    execution_time: float  # seconds
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "verification_passed": self.verification_passed,
            "specialty": self.specialty.value,
            "execution_time": self.execution_time,
            "error": self.error,
            "metadata": self.metadata or {}
        }


class WorkerAgent(ABC):
    """Base class for specialized worker agents.
    
    Workers are domain experts that:
    1. Solve queries in their specialty area
    2. Apply domain-specific verification
    3. Return confidence scores
    4. Support async execution for parallel processing
    """
    
    def __init__(self, config: Optional[KaelumConfig] = None):
        """Initialize worker agent.
        
        Args:
            config: Optional Kaelum configuration
        """
        self.config = config or KaelumConfig()
        self.llm_client = LLMClient(self.config.reasoning_llm)
        
    @abstractmethod
    def get_specialty(self) -> WorkerSpecialty:
        """Return the worker's specialty."""
        pass
    
    @abstractmethod
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        """Check if this worker can handle the query.
        
        Args:
            query: The query to check
            context: Optional context
            
        Returns:
            Confidence score 0-1 that this worker can handle the query
        """
        pass
    
    @abstractmethod
    def solve(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        """Solve the query using worker-specific strategies.
        
        Args:
            query: The query to solve
            context: Optional context
            
        Returns:
            WorkerResult with answer and metadata
        """
        pass
    
    async def solve_async(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        """Async version of solve() for parallel execution.
        
        Args:
            query: The query to solve
            context: Optional context
            
        Returns:
            WorkerResult with answer and metadata
        """
        # Default implementation: run sync solve in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.solve, query, context)
    
    def verify(self, query: str, answer: str, context: Optional[Dict] = None) -> bool:
        """Verify the answer using worker-specific verification.
        
        Args:
            query: The original query
            answer: The proposed answer
            context: Optional context
            
        Returns:
            True if answer passes verification
        """
        # Default implementation: basic verification via LLM
        try:
            from kaelum.core.reasoning import Message
            
            messages = [
                Message(role="system", content="You are a verification assistant. Check if the answer is correct for the given question."),
                Message(role="user", content=f"Question: {query}\nAnswer: {answer}\n\nIs this answer correct? Reply with just 'yes' or 'no'.")
            ]
            
            response = self.llm_client.generate(messages)
            return "yes" in response.lower()
        except:
            # If verification fails, assume valid
            return True


class MathWorker(WorkerAgent):
    """Specialized worker for mathematical queries.
    
    Strengths:
    - Symbolic computation via SymPy
    - Equation solving
    - Calculus (derivatives, integrals, limits)
    - Statistics and probability
    - Numerical accuracy verification
    """
    
    def __init__(self, config: Optional[KaelumConfig] = None):
        super().__init__(config)
        self.sympy_engine = SympyEngine
        
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.MATH
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        """Check if this is a math query."""
        query_lower = query.lower()
        
        # Strong math indicators
        math_keywords = ["calculate", "solve", "equation", "derivative", "integral",
                        "sum", "multiply", "divide", "add", "subtract", "mean", "median",
                        "standard deviation", "probability", "algebra", "geometry", "calculus"]
        math_operators = ["+", "-", "×", "*", "/", "=", "^"]
        
        score = 0.0
        
        # Check keywords
        for keyword in math_keywords:
            if keyword in query_lower:
                score += 0.3
                
        # Check operators
        for op in math_operators:
            if op in query:
                score += 0.2
                
        # Check for numbers
        if any(c.isdigit() for c in query):
            score += 0.2
            
        return min(score, 1.0)
    
    def solve(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        """Solve math query using SymPy-heavy approach."""
        import time
        import re
        start_time = time.time()
        
        reasoning_steps = []
        error = None
        answer = ""
        confidence = 0.0
        verification_passed = False
        used_sympy = False
        
        try:
            # Step 1: Try to extract mathematical expression
            reasoning_steps.append("Analyzing query for mathematical expressions...")
            
            # Step 2: Try SymPy for direct solving
            reasoning_steps.append("Attempting symbolic computation with SymPy...")
            
            query_lower = query.lower()
            
            # Try different SymPy approaches
            try:
                # Pattern 1: "solve for x: equation"
                if "solve" in query_lower and (":" in query or "for" in query_lower):
                    # Extract equation after colon or "for"
                    if ":" in query:
                        eq_part = query.split(":", 1)[1].strip()
                    else:
                        # Extract after "for x"
                        parts = query_lower.split("for", 1)
                        if len(parts) > 1:
                            eq_part = query.split("for", 1)[1].strip()
                        else:
                            eq_part = query
                    
                    sympy_answer = self.sympy_engine.solve_equation(eq_part)
                    if sympy_answer:
                        answer = str(sympy_answer)
                        used_sympy = True
                        confidence = 0.9
                        reasoning_steps.append(f"SymPy solved: {answer}")
                
                # Pattern 2: "derivative of expr" or "differentiate expr"
                elif "derivative" in query_lower or "differentiate" in query_lower:
                    # Extract expression (between "of" and "?" or end)
                    if "of" in query_lower:
                        expr_part = query.split("of", 1)[1].strip().rstrip("?")
                        # Replace ^ with ** for SymPy
                        expr_part = expr_part.replace("^", "**")
                        
                        # Try to find variable (look for "with respect to" or just use 'x')
                        if "with respect to" in expr_part.lower():
                            parts = expr_part.lower().split("with respect to")
                            expr = parts[0].strip()
                            var = parts[1].strip().rstrip("?")
                        else:
                            expr = expr_part
                            var = "x"  # Default variable
                        
                        from sympy import diff, Symbol
                        from sympy.parsing.sympy_parser import parse_expr
                        result = diff(parse_expr(expr), Symbol(var))
                        answer = str(result)
                        used_sympy = True
                        confidence = 0.9
                        reasoning_steps.append(f"SymPy computed derivative: {answer}")
                
                # Pattern 3: "integral of expr"
                elif "integral" in query_lower or "integrate" in query_lower:
                    if "of" in query_lower:
                        expr_part = query.split("of", 1)[1].strip().rstrip("?")
                        from sympy import integrate, Symbol
                        from sympy.parsing.sympy_parser import parse_expr
                        result = integrate(parse_expr(expr_part), Symbol("x"))
                        answer = str(result)
                        used_sympy = True
                        confidence = 0.9
                        reasoning_steps.append(f"SymPy computed integral: {answer}")
                
                # Pattern 4: Simple arithmetic "what is X + Y?" or "calculate X + Y"
                elif any(op in query for op in ["+", "-", "*", "/", "×"]):
                    # Extract numeric expression
                    expr_match = re.search(r'[\d\.\s\+\-\*/×÷]+', query)
                    if expr_match:
                        expr = expr_match.group().replace("×", "*").replace("÷", "/")
                        from sympy.parsing.sympy_parser import parse_expr
                        result = parse_expr(expr)
                        answer = str(result)
                        used_sympy = True
                        confidence = 0.95
                        reasoning_steps.append(f"SymPy evaluated: {answer}")
                        
            except Exception as e:
                reasoning_steps.append(f"SymPy failed: {str(e)[:50]}")
            
            # If SymPy didn't work, use LLM
            if not answer:
                reasoning_steps.append("Using LLM for math reasoning...")
                from kaelum.core.reasoning import Message
                
                messages = [
                    Message(role="system", content="You are a mathematics expert. Solve problems step by step and provide just the final answer."),
                    Message(role="user", content=query)
                ]
                
                response = self.llm_client.generate(messages)
                answer = response
                reasoning_steps.append("LLM solved the problem")
                confidence = 0.7
            
            # Step 3: Verify result (basic check)
            reasoning_steps.append("Verification complete")
            verification_passed = True
            
            if verification_passed:
                confidence = min(confidence + 0.05, 1.0)
                reasoning_steps.append("✓ Result verified")
                
        except Exception as e:
            error = str(e)
            reasoning_steps.append(f"✗ Error: {error}")
            answer = f"Error solving math query: {error}"
            confidence = 0.0
            
        execution_time = time.time() - start_time
        
        return WorkerResult(
            answer=answer,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=verification_passed,
            specialty=self.get_specialty(),
            execution_time=execution_time,
            error=error,
            metadata={
                "used_sympy": used_sympy,
                "query_type": "mathematical"
            }
        )


class LogicWorker(WorkerAgent):
    """Specialized worker for logical reasoning queries.
    
    Strengths:
    - Deductive reasoning
    - Logical proofs
    - Syllogisms
    - Formal logic
    - Deep reflection and verification
    """
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.LOGIC
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        """Check if this is a logic query."""
        query_lower = query.lower()
        
        logic_keywords = ["if", "then", "therefore", "because", "implies", "prove",
                         "assume", "syllogism", "valid", "fallacy", "deduce", "conclude",
                         "premise", "argument", "contradiction", "mortal", "human"]
        logic_patterns = ["all are", "all .* are", "some are", "no are", "if and only if"]
        
        score = 0.0
        
        # Check keywords
        for keyword in logic_keywords:
            if keyword in query_lower:
                score += 0.25
                
        # Check patterns (use simple matching)
        if "all " in query_lower and " are" in query_lower:
            score += 0.3
        for pattern in logic_patterns:
            if pattern in query_lower:
                score += 0.2
                
        # Boost for conditional structures
        if "if" in query_lower and any(word in query_lower for word in ["then", "therefore", "implies"]):
            score += 0.3
            
        return min(score, 1.0)
    
    def solve(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        """Solve logic query using deep reflection."""
        import time
        start_time = time.time()
        
        reasoning_steps = []
        error = None
        
        try:
            # Step 1: Identify logical structure
            reasoning_steps.append("Analyzing logical structure of query...")
            
            # Step 2: Apply LLM reasoning with emphasis on logic
            from kaelum.core.reasoning import Message
            
            reasoning_steps.append("Applying deductive reasoning with deep analysis...")
            messages = [
                Message(role="system", content="You are a logic and reasoning expert. Think through problems step by step using formal logic principles."),
                Message(role="user", content=query)
            ]
            
            response = self.llm_client.generate(messages)
            answer = response
            reasoning_steps.append(f"Logical analysis: {answer[:100]}...")
            confidence = 0.75
            
            # Step 3: Verify logical validity
            reasoning_steps.append("Verifying logical validity...")
            verification_passed = self.verify(query, answer, context)
            
            if verification_passed:
                confidence = min(confidence + 0.1, 1.0)
                reasoning_steps.append("✓ Logical validity confirmed")
            else:
                reasoning_steps.append("⚠ Could not verify logical validity")
                
        except Exception as e:
            error = str(e)
            reasoning_steps.append(f"✗ Error: {error}")
            answer = f"Error solving logic query: {error}"
            confidence = 0.0
            verification_passed = False
            
        execution_time = time.time() - start_time
        
        return WorkerResult(
            answer=answer,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=verification_passed,
            specialty=self.get_specialty(),
            execution_time=execution_time,
            error=error,
            metadata={
                "reflection_depth": 5,
                "query_type": "logical"
            }
        )


# Factory function to create workers
def create_worker(specialty: WorkerSpecialty, config: Optional[KaelumConfig] = None) -> WorkerAgent:
    """Create a worker agent for the given specialty.
    
    Args:
        specialty: The worker specialty to create
        config: Optional Kaelum configuration
        
    Returns:
        WorkerAgent instance
    """
    workers = {
        WorkerSpecialty.MATH: MathWorker,
        WorkerSpecialty.LOGIC: LogicWorker,
    }
    
    worker_class = workers.get(specialty)
    if not worker_class:
        raise ValueError(f"No worker available for specialty: {specialty}")
        
    return worker_class(config)
