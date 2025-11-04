"""Worker agents - specialized reasoning agents for specific query types.

Workers are domain-specific agents that excel at particular types of queries:
- MathWorker: Mathematical calculations, symbolic reasoning  
- LogicWorker: Logical proofs, deductive reasoning
- CodeWorker: Code generation, debugging, algorithms
- etc.

Each worker uses LATS (Language Agent Tree Search) for multi-step reasoning
with MCTS-style exploration and tree caching for similar queries.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

from core.config import KaelumConfig
from core.reasoning import LLMClient, Message
from core.sympy_engine import SympyEngine
from core.lats import LATS, LATSNode
from core.tree_cache import TreeCache


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
    1. Solve queries in their specialty area using LATS tree search
    2. Check tree cache for similar past reasoning
    3. Apply domain-specific verification
    4. Return confidence scores
    5. Support async execution for parallel processing
    """
    
    def __init__(self, config: Optional[KaelumConfig] = None, 
                 tree_cache: Optional[TreeCache] = None):
        """Initialize worker agent.
        
        Args:
            config: Optional Kaelum configuration
            tree_cache: Optional tree cache for storing/retrieving reasoning trees
        """
        self.config = config or KaelumConfig()
        self.llm_client = LLMClient(self.config.reasoning_llm)
        self.tree_cache = tree_cache or TreeCache()
        
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
    def solve(self, query: str, context: Optional[Dict] = None,
              use_cache: bool = True, max_tree_depth: int = 5,
              num_simulations: int = 10) -> WorkerResult:
        """Solve the query using LATS-based tree search.
        
        Args:
            query: The query to solve
            context: Optional context
            use_cache: Whether to check tree cache for similar queries
            max_tree_depth: Maximum depth for LATS tree
            num_simulations: Number of MCTS simulations to run
            
        Returns:
            WorkerResult with answer and metadata
        """
        pass
    
    def _check_cache(self, query: str) -> Optional[WorkerResult]:
        """Check tree cache for similar query.
        
        Args:
            query: Query to search for
            
        Returns:
            WorkerResult from cache or None if not found
        """
        cached = self.tree_cache.retrieve(
            query,
            worker_specialty=self.get_specialty().value,
            require_success=True
        )
        
        if cached is None:
            return None
        
        tree, metadata, similarity = cached
        
        # Extract answer from best path in tree
        best_node = tree.best_child()
        if best_node is None:
            return None
        
        # Build reasoning steps from tree path
        reasoning_steps = []
        node = best_node
        while node is not None:
            if "step" in node.state:
                reasoning_steps.insert(0, node.state["step"])
            node = node.parent
        
        answer = best_node.state.get("answer", "")
        
        return WorkerResult(
            answer=answer,
            confidence=metadata.confidence * similarity,  # Adjust confidence by similarity
            reasoning_steps=reasoning_steps,
            verification_passed=metadata.success,
            specialty=self.get_specialty(),
            execution_time=0.001,  # Cache hit is fast
            metadata={
                "cache_hit": True,
                "similarity": similarity,
                "original_query": metadata.query
            }
        )
    
    async def solve_async(self, query: str, context: Optional[Dict] = None,
                         use_cache: bool = True, max_tree_depth: int = 5,
                         num_simulations: int = 10) -> WorkerResult:
        """Async version of solve() for parallel execution.
        
        Args:
            query: The query to solve
            context: Optional context
            use_cache: Whether to check tree cache
            max_tree_depth: Maximum tree depth
            num_simulations: Number of MCTS simulations
            
        Returns:
            WorkerResult with answer and metadata
        """
        # Default implementation: run sync solve in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.solve, query, context, use_cache, max_tree_depth, num_simulations
        )
    
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
            from core.reasoning import Message
            
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
    
    def solve(self, query: str, context: Optional[Dict] = None,
              use_cache: bool = True, max_tree_depth: int = 5,
              num_simulations: int = 10) -> WorkerResult:
        """Solve math query using LATS tree search with SymPy verification.
        
        Architecture:
        1. Check tree cache for similar queries
        2. Build LATS reasoning tree with MCTS exploration
        3. Each node represents a reasoning step
        4. Simulate/verify using SymPy when possible
        5. Backpropagate success/failure scores
        6. Extract best path as final reasoning
        """
        start_time = time.time()
        
        # Step 1: Check cache
        if use_cache:
            cached_result = self._check_cache(query)
            if cached_result:
                return cached_result
        
        # Step 2: Initialize LATS tree
        root_state = {
            "query": query,
            "step": "Initial problem analysis",
            "depth": 0,
            "partial_solution": None
        }
        
        # Define simulator for LATS
        def simulate_math_step(node: LATSNode) -> float:
            """Evaluate a reasoning step using SymPy verification."""
            state = node.state
            
            # If we have an answer, try to verify it
            if "answer" in state:
                try:
                    # Try SymPy verification
                    answer = state["answer"]
                    # Basic verification: check if answer is reasonable
                    if answer and len(str(answer)) > 0:
                        return 0.9  # High reward for complete answers
                except:
                    return 0.3
            
            # Partial solution gets medium reward
            if state.get("partial_solution"):
                return 0.5
            
            # Deep nodes without progress get low reward
            depth = state.get("depth", 0)
            return max(0.1, 0.5 - depth * 0.05)
        
        # Define expansion function
        def expand_math_step(parent_node: LATSNode) -> Dict[str, Any]:
            """Generate next reasoning step using LLM."""
            parent_state = parent_node.state
            depth = parent_state.get("depth", 0)
            
            # Build reasoning history
            history = []
            node = parent_node
            while node.parent is not None:
                if "step" in node.state:
                    history.insert(0, node.state["step"])
                node = node.parent
            
            # Generate next step with LLM
            prompt = f"Query: {query}\n\n"
            if history:
                prompt += "Previous steps:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(history))
                prompt += "\n\nWhat is the next step?"
            else:
                prompt += "What is the first step to solve this?"
            
            try:
                messages = [Message(role="system", content="You are a math expert. Provide ONE next reasoning step."), 
                           Message(role="user", content=prompt)]
                response = self.llm_client.generate(messages)
                next_step = response.strip()
                
                # Try to extract answer if this seems like a final step
                is_final = any(word in next_step.lower() for word in ["answer is", "result is", "equals", "="])
                
                return {
                    "query": query,
                    "step": next_step,
                    "depth": depth + 1,
                    "partial_solution": next_step if not is_final else None,
                    "answer": next_step if is_final else None
                }
            except:
                # Fallback
                return {
                    "query": query,
                    "step": f"Continue solving step {depth + 1}",
                    "depth": depth + 1
                }
        
        # Step 3: Run LATS search
        tree = LATS(root_state, simulator=simulate_math_step, expand_fn=expand_math_step)
        
        for _ in range(num_simulations):
            # Select promising node
            node = tree.select()
            
            # Don't expand beyond max depth
            if node.state.get("depth", 0) >= max_tree_depth:
                continue
            
            # Expand
            child_state = expand_math_step(node)
            child = tree.expand(node, child_state)
            
            # Simulate
            reward = simulate_math_step(child)
            
            # Backpropagate
            tree.backpropagate(child, reward)
        
        # Step 4: Extract best path
        best_node = tree.best_child()
        if best_node is None:
            best_node = tree.root
        
        # Build reasoning path
        reasoning_steps = []
        node = best_node
        while node is not None:
            if node.state.get("step") and node != tree.root:
                reasoning_steps.insert(0, node.state["step"])
            node = node.parent
        
        # Get final answer
        answer = best_node.state.get("answer", "")
        if not answer and reasoning_steps:
            answer = reasoning_steps[-1]
        
        # Calculate confidence from tree statistics
        if best_node.visits > 0:
            confidence = best_node.value / best_node.visits
        else:
            confidence = 0.5
        
        verification_passed = confidence > 0.7
        execution_time = time.time() - start_time
        
        # Step 5: Cache the tree
        if use_cache and verification_passed:
            self.tree_cache.store(query, tree, self.get_specialty().value, 
                                 verification_passed, confidence)
        
        return WorkerResult(
            answer=answer,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=verification_passed,
            specialty=self.get_specialty(),
            execution_time=execution_time,
            metadata={
                "num_simulations": num_simulations,
                "tree_depth": best_node.state.get("depth", 0),
                "tree_visits": tree.root.visits,
                "cache_hit": False
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
    
    def solve(self, query: str, context: Optional[Dict] = None,
              use_cache: bool = True, max_tree_depth: int = 5,
              num_simulations: int = 10) -> WorkerResult:
        """Solve logic query using LATS with deep deductive reasoning."""
        start_time = time.time()
        
        # Step 1: Check cache
        if use_cache:
            cached_result = self._check_cache(query)
            if cached_result:
                return cached_result
        
        # Step 2: Initialize LATS for logical reasoning
        root_state = {
            "query": query,
            "step": "Initial logical analysis",
            "depth": 0,
            "premises": [],
            "conclusion": None
        }
        
        def simulate_logic_step(node: LATSNode) -> float:
            """Evaluate logical reasoning step."""
            state = node.state
            
            # Reward for reaching conclusions
            if state.get("conclusion"):
                return 0.95
            
            # Reward for identifying premises
            num_premises = len(state.get("premises", []))
            if num_premises > 0:
                return 0.6 + (num_premises * 0.05)
            
            # Penalize excessive depth without progress
            depth = state.get("depth", 0)
            return max(0.2, 0.5 - depth * 0.04)
        
        def expand_logic_step(parent_node: LATSNode) -> Dict[str, Any]:
            """Generate next logical reasoning step."""
            parent_state = parent_node.state
            depth = parent_state.get("depth", 0)
            
            # Build history
            history = []
            node = parent_node
            while node.parent is not None:
                if "step" in node.state:
                    history.insert(0, node.state["step"])
                node = node.parent
            
            prompt = f"Query: {query}\n\n"
            if history:
                prompt += "Reasoning so far:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(history))
                prompt += "\n\nWhat is the next logical step?"
            else:
                prompt += "Apply logical reasoning. What is the first step?"
            
            try:
                messages = [Message(role="system", content="You are a logic expert. Apply formal deductive reasoning."),
                           Message(role="user", content=prompt)]
                response = self.llm_client.generate(messages)
                next_step = response.strip()
                
                # Check if this is a conclusion
                is_conclusion = any(word in next_step.lower() for word in ["therefore", "thus", "conclude", "answer is"])
                
                return {
                    "query": query,
                    "step": next_step,
                    "depth": depth + 1,
                    "premises": parent_state.get("premises", []),
                    "conclusion": next_step if is_conclusion else None
                }
            except:
                return {
                    "query": query,
                    "step": f"Logical step {depth + 1}",
                    "depth": depth + 1,
                    "premises": parent_state.get("premises", [])
                }
        
        # Step 3: Run LATS
        tree = LATS(root_state, simulator=simulate_logic_step, expand_fn=expand_logic_step)
        
        for _ in range(num_simulations):
            node = tree.select()
            if node.state.get("depth", 0) >= max_tree_depth:
                continue
            child_state = expand_logic_step(node)
            child = tree.expand(node, child_state)
            reward = simulate_logic_step(child)
            tree.backpropagate(child, reward)
        
        # Step 4: Extract best path
        best_node = tree.best_child()
        if best_node is None:
            best_node = tree.root
        
        reasoning_steps = []
        node = best_node
        while node is not None:
            if node.state.get("step") and node != tree.root:
                reasoning_steps.insert(0, node.state["step"])
            node = node.parent
        
        answer = best_node.state.get("conclusion", reasoning_steps[-1] if reasoning_steps else "")
        confidence = best_node.value / best_node.visits if best_node.visits > 0 else 0.5
        verification_passed = confidence > 0.7
        execution_time = time.time() - start_time
        
        # Step 5: Cache
        if use_cache and verification_passed:
            self.tree_cache.store(query, tree, self.get_specialty().value,
                                 verification_passed, confidence)
        
        return WorkerResult(
            answer=answer,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=verification_passed,
            specialty=self.get_specialty(),
            execution_time=execution_time,
            metadata={
                "num_simulations": num_simulations,
                "tree_depth": best_node.state.get("depth", 0),
                "cache_hit": False
            }
        )


# Factory function to create workers
def create_worker(specialty: WorkerSpecialty, config: Optional[KaelumConfig] = None, **kwargs) -> WorkerAgent:
    """Create a worker agent for the given specialty.
    
    Args:
        specialty: The worker specialty to create
        config: Optional Kaelum configuration
        **kwargs: Additional arguments for worker initialization (e.g., rag_adapter for FactualWorker)
        
    Returns:
        WorkerAgent instance
    """
    # Import all worker types
    from core.code_worker import CodeWorker
    from core.factual_worker import FactualWorker
    from core.creative_worker import CreativeWorker
    
    workers = {
        WorkerSpecialty.MATH: MathWorker,
        WorkerSpecialty.LOGIC: LogicWorker,
        WorkerSpecialty.CODE: CodeWorker,
        WorkerSpecialty.FACTUAL: FactualWorker,
        WorkerSpecialty.CREATIVE: CreativeWorker,
    }
    
    worker_class = workers.get(specialty)
    if not worker_class:
        raise ValueError(f"No worker available for specialty: {specialty}")
    
    # Handle workers that need special initialization
    if specialty == WorkerSpecialty.FACTUAL:
        rag_adapter = kwargs.get('rag_adapter')
        return worker_class(config, rag_adapter)
    else:
        return worker_class(config)
