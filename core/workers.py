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
    MATH = "math"
    LOGIC = "logic"
    CODE = "code"
    FACTUAL = "factual"
    CREATIVE = "creative"
    ANALYSIS = "analysis"


@dataclass
class WorkerResult:
    answer: str
    confidence: float
    reasoning_steps: List[str]
    verification_passed: bool
    specialty: WorkerSpecialty
    execution_time: float
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
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
    
    def __init__(self, config: Optional[KaelumConfig] = None, 
                 tree_cache: Optional[TreeCache] = None):
        self.config = config or KaelumConfig()
        self.llm_client = LLMClient(self.config.reasoning_llm)
        self.tree_cache = tree_cache or TreeCache()
        
    @abstractmethod
    def get_specialty(self) -> WorkerSpecialty:
        pass
    
    @abstractmethod
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        pass
    
    @abstractmethod
    def solve(self, query: str, context: Optional[Dict] = None,
              use_cache: bool = True, max_tree_depth: int = 5,
              num_simulations: int = 10) -> WorkerResult:
        pass
    
    def _check_cache(self, query: str) -> Optional[WorkerResult]:
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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.solve, query, context, use_cache, max_tree_depth, num_simulations
        )
    
    def verify(self, query: str, answer: str, context: Optional[Dict] = None) -> bool:
        try:
            from core.reasoning import Message
            
            messages = [
                Message(role="system", content="You are a verification assistant. Check if the answer is correct for the given question."),
                Message(role="user", content=f"Question: {query}\nAnswer: {answer}\n\nIs this answer correct? Reply with just 'yes' or 'no'.")
            ]
            
            response = self.llm_client.generate(messages)
            return "yes" in response.lower()
        except:
            return True


class MathWorker(WorkerAgent):
    
    def __init__(self, config: Optional[KaelumConfig] = None):
        super().__init__(config)
        self.sympy_engine = SympyEngine
        
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.MATH
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        query_lower = query.lower()
        
        math_keywords = ["calculate", "solve", "equation", "derivative", "integral",
                        "sum", "multiply", "divide", "add", "subtract", "mean", "median",
                        "standard deviation", "probability", "algebra", "geometry", "calculus"]
        math_operators = ["+", "-", "×", "*", "/", "=", "^"]
        
        score = 0.0
        
        for keyword in math_keywords:
            if keyword in query_lower:
                score += 0.3
                
        for op in math_operators:
            if op in query:
                score += 0.2
                
        if any(c.isdigit() for c in query):
            score += 0.2
            
        return min(score, 1.0)
    
    def solve(self, query: str, context: Optional[Dict] = None,
              use_cache: bool = True, max_tree_depth: int = 5,
              num_simulations: int = 10) -> WorkerResult:
        start_time = time.time()
        
        if use_cache:
            cached_result = self._check_cache(query)
            if cached_result:
                return cached_result
        
        root_state = {
            "query": query,
            "step": "Initial problem analysis",
            "depth": 0,
            "partial_solution": None
        }
        
        def simulate_math_step(node: LATSNode) -> float:
            state = node.state
            
            if "answer" in state:
                try:
                    answer = state["answer"]
                    if answer and len(str(answer)) > 0:
                        return 0.9
                except:
                    return 0.3
            
            if state.get("partial_solution"):
                return 0.5
            
            depth = state.get("depth", 0)
            return max(0.1, 0.5 - depth * 0.05)
        
        def expand_math_step(parent_node: LATSNode) -> Dict[str, Any]:
            parent_state = parent_node.state
            depth = parent_state.get("depth", 0)
            
            history = []
            node = parent_node
            while node.parent is not None:
                if "step" in node.state:
                    history.insert(0, node.state["step"])
                node = node.parent
            
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
                
                is_final = any(word in next_step.lower() for word in ["answer is", "result is", "equals", "="])
                
                return {
                    "query": query,
                    "step": next_step,
                    "depth": depth + 1,
                    "partial_solution": next_step if not is_final else None,
                    "answer": next_step if is_final else None
                }
            except:
                return {
                    "query": query,
                    "step": f"Continue solving step {depth + 1}",
                    "depth": depth + 1
                }
        
        tree = LATS(root_state, simulator=simulate_math_step, expand_fn=expand_math_step)
        
        for _ in range(num_simulations):
            node = tree.select()
            
            if node.state.get("depth", 0) >= max_tree_depth:
                continue
            
            child_state = expand_math_step(node)
            child = tree.expand(node, child_state)
            
            reward = simulate_math_step(child)
            
            tree.backpropagate(child, reward)
        
        best_node = tree.best_child()
        if best_node is None:
            best_node = tree.root
        
        reasoning_steps = []
        node = best_node
        while node is not None:
            if node.state.get("step") and node != tree.root:
                reasoning_steps.insert(0, node.state["step"])
            node = node.parent
        
        answer = best_node.state.get("answer", "")
        if not answer and reasoning_steps:
            answer = reasoning_steps[-1]
        
        if best_node.visits > 0:
            confidence = best_node.value / best_node.visits
        else:
            confidence = 0.5
        
        verification_passed = confidence > 0.7
        execution_time = time.time() - start_time
        
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
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.LOGIC
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        query_lower = query.lower()
        
        logic_keywords = ["if", "then", "therefore", "because", "implies", "prove",
                         "assume", "syllogism", "valid", "fallacy", "deduce", "conclude",
                         "premise", "argument", "contradiction", "mortal", "human"]
        logic_patterns = ["all are", "all .* are", "some are", "no are", "if and only if"]
        
        score = 0.0
        
        for keyword in logic_keywords:
            if keyword in query_lower:
                score += 0.25
                
        if "all " in query_lower and " are" in query_lower:
            score += 0.3
        for pattern in logic_patterns:
            if pattern in query_lower:
                score += 0.2
                
        if "if" in query_lower and any(word in query_lower for word in ["then", "therefore", "implies"]):
            score += 0.3
            
        return min(score, 1.0)
    
    def solve(self, query: str, context: Optional[Dict] = None,
              use_cache: bool = True, max_tree_depth: int = 5,
              num_simulations: int = 10) -> WorkerResult:
        start_time = time.time()
        
        if use_cache:
            cached_result = self._check_cache(query)
            if cached_result:
                return cached_result
        
        root_state = {
            "query": query,
            "step": "Initial logical analysis",
            "depth": 0,
            "premises": [],
            "conclusion": None
        }
        
        def simulate_logic_step(node: LATSNode) -> float:
            state = node.state
            
            if state.get("conclusion"):
                return 0.95
            
            num_premises = len(state.get("premises", []))
            if num_premises > 0:
                return 0.6 + (num_premises * 0.05)
            
            depth = state.get("depth", 0)
            return max(0.2, 0.5 - depth * 0.04)
        
        def expand_logic_step(parent_node: LATSNode) -> Dict[str, Any]:
            parent_state = parent_node.state
            depth = parent_state.get("depth", 0)
            
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
        
        tree = LATS(root_state, simulator=simulate_logic_step, expand_fn=expand_logic_step)
        
        for _ in range(num_simulations):
            node = tree.select()
            if node.state.get("depth", 0) >= max_tree_depth:
                continue
            child_state = expand_logic_step(node)
            child = tree.expand(node, child_state)
            reward = simulate_logic_step(child)
            tree.backpropagate(child, reward)
        
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


def create_worker(specialty: WorkerSpecialty, config: Optional[KaelumConfig] = None, **kwargs) -> WorkerAgent:
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
    
    return worker_class(config)
