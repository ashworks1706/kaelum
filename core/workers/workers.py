import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger("kaelum.worker")

from ..config import KaelumConfig
from ..reasoning import LLMClient, Message
from ..verification import SympyEngine
from ..search import LATS, LATSNode
from ..search import TreeCache
from ..search import RewardModel
from ..detectors import ConclusionDetector
from ..learning import AdaptivePenalty

class WorkerSpecialty(Enum):
    MATH = "math"
    LOGIC = "logic"
    CODE = "code"
    FACTUAL = "factual"
    CREATIVE = "creative"
    ANALYSIS = "analysis"

@dataclass
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
    lats_tree: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "verification_passed": self.verification_passed,
            "specialty": self.specialty.value,
            "execution_time": self.execution_time,
            "error": self.error,
            "metadata": self.metadata or {}
        }
        
        if self.lats_tree is not None:
            result["lats_tree"] = self.lats_tree.root.to_dict()
        
        return result

class WorkerAgent(ABC):
    
    def __init__(self, config: Optional[KaelumConfig] = None, 
                 tree_cache: Optional[TreeCache] = None):
        self.config = config or KaelumConfig()
        self.llm_client = LLMClient(self.config.reasoning_llm)
        self.tree_cache = tree_cache
        self.lats_params = {
            "exploration_constant": getattr(self.config, "lats_exploration_constant", 1.414),
            "prune_visit_threshold": getattr(self.config, "lats_prune_visit_threshold", 3),
            "prune_reward_threshold": getattr(self.config, "lats_prune_reward_threshold", 0.3),
        }
    
    def _lightweight_coherence_check(self, node: LATSNode) -> bool:
        state = node.state
        step_text = state.get("step", "")
        
        if not step_text or len(step_text) < 10:
            return False
        
        step_lower = step_text.lower()
        
        incoherent_patterns = [
            "i don't know", "i'm not sure", "unclear", "doesn't make sense",
            "error", "failed", "cannot determine", "impossible to",
            "contradicts", "inconsistent with"
        ]
        
        for pattern in incoherent_patterns:
            if pattern in step_lower:
                return False
        
        if parent := node.parent:
            parent_step = parent.state.get("step", "")
            if parent_step and step_text == parent_step:
                return False
        
        sentences = [s.strip() for s in step_text.split('.') if s.strip()]
        if len(sentences) > 1:
            first_words = set()
            for sent in sentences:
                words = sent.lower().split()
                if words and words[0] in first_words:
                    return False
                if words:
                    first_words.add(words[0])
        
        return True
        
    @abstractmethod
    def get_specialty(self) -> WorkerSpecialty:
        pass
    
    def get_system_prompt(self) -> str:
        specialty_map = {
            WorkerSpecialty.MATH: self.config.worker_prompts.math,
            WorkerSpecialty.LOGIC: self.config.worker_prompts.logic,
            WorkerSpecialty.CODE: self.config.worker_prompts.code,
            WorkerSpecialty.FACTUAL: self.config.worker_prompts.factual,
            WorkerSpecialty.CREATIVE: self.config.worker_prompts.creative,
            WorkerSpecialty.ANALYSIS: self.config.worker_prompts.analysis
        }
        return specialty_map.get(self.get_specialty(), "You are a helpful reasoning assistant.")
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        return 1.0
    
    @abstractmethod
    def solve(self, query: str, context: Optional[Dict] = None,
              use_cache: bool = True, max_tree_depth: int = 5,
              num_simulations: int = 10, parallel: bool = False,
              max_workers: int = 4) -> WorkerResult:
        pass
    
    def _check_cache(self, query: str) -> Optional[WorkerResult]:
        if not self.tree_cache:
            return None
            
        cached = self.tree_cache.retrieve(
            query,
            worker_specialty=None,
            require_success=True
        )
        
        if cached is None:
            return None
        
        tree, metadata, similarity = cached
        
        best_node = tree.best_child()
        if best_node is None:
            return None
        
        reasoning_steps = []
        node = best_node
        while node is not None:
            if "step" in node.state:
                reasoning_steps.insert(0, node.state["step"])
            node = node.parent
        
        answer = best_node.state.get("answer", "")
        
        return WorkerResult(
            answer=answer,
            confidence=metadata.confidence * similarity,
            reasoning_steps=reasoning_steps,
            verification_passed=metadata.success,
            specialty=self.get_specialty(),
            execution_time=0.001,
            metadata={
                "cache_hit": True,
                "similarity": similarity,
                "original_query": metadata.query
            }
        )

    def _build_lats(self, root_state, simulator, expand_fn, coherence_checker=None):
        return LATS(
            root_state,
            simulator=simulator,
            expand_fn=expand_fn,
            coherence_checker=coherence_checker or self._lightweight_coherence_check,
            exploration_constant=self.lats_params["exploration_constant"],
            prune_visit_threshold=self.lats_params["prune_visit_threshold"],
            prune_reward_threshold=self.lats_params["prune_reward_threshold"],
        )
    
    async def solve_async(self, query: str, context: Optional[Dict] = None,
                         use_cache: bool = True, max_tree_depth: int = 5,
                         num_simulations: int = 10, parallel: bool = False) -> WorkerResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.solve, query, context, use_cache, max_tree_depth, num_simulations
        )

    def _penalize_failed_path(
        self,
        tree,
        issues: List[str],
        penalty: float = 0.5,
    ) -> None:
        """Penalise the current best path so MCTS explores alternatives next.

        Called when verification failed on the previous iteration.  We subtract
        ``penalty`` from every node on the best-child chain so the UCT scores
        push subsequent simulations toward unexplored branches.  Nodes that
        were pruned but have few visits are also un-pruned so they remain
        candidates for expansion.
        """
        penalised = 0
        node = tree.best_child()
        while node and node.parent is not None:
            node.value = max(-1.0, node.value - penalty)
            node.pruned = False      # ensure parent can still reach this node
            penalised += 1
            node = node.parent

        # Un-prune lightly-explored nodes outside the failed path so MCTS can
        # still visit them in subsequent simulations.
        for n in tree.nodes.values():
            if n.pruned and n.visits <= 1:
                n.pruned = False

        logger.info(
            f"TREE-REUSE: Penalised {penalised} nodes on failed path; "
            f"{len(tree.nodes)} total nodes available for continued search"
        )

class MathWorker(WorkerAgent):
    
    def __init__(self, config: Optional[KaelumConfig] = None, tree_cache: Optional[TreeCache] = None):
        super().__init__(config, tree_cache)
        self.conclusion_detector = ConclusionDetector(embedding_model=self.config.embedding_model)
        
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.MATH
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        return 1.0
    
    def solve(self, query: str, context: Optional[Dict] = None,
              use_cache: bool = True, max_tree_depth: int = 5,
              num_simulations: int = 10, parallel: bool = False,
              max_workers: int = 4,
              existing_tree=None,
              extra_sims: int = 0,
              verification_issues: Optional[List[str]] = None) -> WorkerResult:
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
            depth = state.get("depth", 0)
            
            if "answer" in state:
                answer = state["answer"]
                has_answer = answer and len(str(answer)) > 0
                query_complexity = AdaptivePenalty.compute_complexity(query)
                return RewardModel.get_reward("math", state, depth, has_answer=has_answer, query_complexity=query_complexity)
            
            has_partial = bool(state.get("partial_solution"))
            query_complexity = AdaptivePenalty.compute_complexity(query)
            return RewardModel.get_reward("math", state, depth, has_partial=has_partial, query_complexity=query_complexity)
        
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
                prompt += "\n\nProvide ONLY the next single step. Keep it concise (1-3 sentences). Do not solve the entire problem."
            else:
                prompt += "Provide ONLY the first step to solve this. Keep it concise (1-3 sentences)."
            
            messages = [
                Message(role="system", content=self.get_system_prompt()), 
                Message(role="user", content=prompt)
            ]
            response = self.llm_client.generate(messages)
            next_step = response.strip()
            
            conclusion_result = self.conclusion_detector.detect(next_step, history)
            is_final = depth >= max_tree_depth - 1 or conclusion_result['is_conclusion']
            
            return {
                "query": query,
                "step": next_step,
                "depth": depth + 1,
                "partial_solution": next_step if not is_final else None,
                "answer": next_step if is_final else None
            }
        if existing_tree is not None:
            tree = existing_tree
            tree.simulator = simulate_math_step
            tree.expand_fn = expand_math_step
            tree.coherence_checker = self._lightweight_coherence_check
            self._penalize_failed_path(tree, verification_issues or [])
            sims = extra_sims if extra_sims > 0 else max(3, num_simulations // 2)
            logger.info(f"TREE-REUSE: Continuing math search ({sims} additional simulations)")
        else:
            tree = self._build_lats(root_state, simulate_math_step, expand_math_step)
            sims = num_simulations
        
        tree.run_simulations(sims, max_tree_depth, parallel=parallel, max_workers=max_workers)
        
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
        
        confidence = RewardModel.compute_confidence(
            best_node.value if best_node else 0.0,
            best_node.visits if best_node else 0,
            tree.root.visits
        )
        
        execution_time = time.time() - start_time
        
        avg_reward = tree.get_avg_reward()
        
        if use_cache:
            self.tree_cache.store(query, tree, self.get_specialty().value, 
                                 True, confidence)
        
        return WorkerResult(
            answer=answer,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=False,
            specialty=self.get_specialty(),
            execution_time=execution_time,
            lats_tree=tree,
            metadata={
                "num_simulations": num_simulations,
                "tree_depth": best_node.state.get("depth", 0) if best_node else 0,
                "tree_visits": tree.root.visits,
                "cache_hit": False,
                "avg_reward": avg_reward
            }
        )

class LogicWorker(WorkerAgent):
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.LOGIC
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        return 1.0
    
    def solve(self, query: str, context: Optional[Dict] = None,
              use_cache: bool = True, max_tree_depth: int = 5,
              num_simulations: int = 10, parallel: bool = False,
              max_workers: int = 4,
              existing_tree=None,
              extra_sims: int = 0,
              verification_issues: Optional[List[str]] = None) -> WorkerResult:
        start_time = time.time()
        
        if use_cache and existing_tree is None:
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
            depth = state.get("depth", 0)
            
            completion = 0.0
            if state.get("conclusion"):
                completion = 1.0
            else:
                num_premises = len(state.get("premises", []))
                completion = min(0.8, num_premises * 0.15)
            
            query_complexity = AdaptivePenalty.compute_complexity(query)
            
            return RewardModel.get_reward("logic", state, depth, completion, 
                                         query_complexity=query_complexity)
        
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
                prompt += "\n\nProvide ONLY the next single logical step. Keep it concise (1-3 sentences). Do not complete the entire reasoning."
            else:
                prompt += "Apply logical reasoning. Provide ONLY the first step (1-3 sentences)."
            
            messages = [
                Message(role="system", content=self.get_system_prompt()),
                Message(role="user", content=prompt)
            ]
            response = self.llm_client.generate(messages)
            next_step = response.strip()
            
            conclusion_result = self.conclusion_detector.detect(next_step, history)
            is_conclusion = depth >= max_tree_depth - 1 or conclusion_result['is_conclusion']
            
            return {
                "query": query,
                "step": next_step,
                "depth": depth + 1,
                "premises": parent_state.get("premises", []),
                "conclusion": next_step if is_conclusion else None
            }
        if existing_tree is not None:
            tree = existing_tree
            tree.simulator = simulate_logic_step
            tree.expand_fn = expand_logic_step
            tree.coherence_checker = self._lightweight_coherence_check
            self._penalize_failed_path(tree, verification_issues or [])
            sims = extra_sims if extra_sims > 0 else max(3, num_simulations // 2)
            logger.info(f"TREE-REUSE: Continuing logic search ({sims} additional simulations)")
        else:
            tree = LATS(root_state, simulator=simulate_logic_step, expand_fn=expand_logic_step,
                        coherence_checker=self._lightweight_coherence_check)
            sims = num_simulations

        tree.run_simulations(sims, max_tree_depth, parallel=parallel, max_workers=max_workers)
        
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
        
        avg_reward = tree.get_avg_reward()
        
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
            lats_tree=tree,
            metadata={
                "num_simulations": num_simulations,
                "tree_depth": best_node.state.get("depth", 0),
                "cache_hit": False,
                "avg_reward": avg_reward
            }
        )

def create_worker(specialty: WorkerSpecialty, config: Optional[KaelumConfig] = None, tree_cache: Optional[TreeCache] = None, **kwargs) -> WorkerAgent:
    from core.workers.code_worker import CodeWorker
    from core.workers.factual_worker import FactualWorker
    from core.workers.creative_worker import CreativeWorker
    from core.workers.analysis_worker import AnalysisWorker
    
    workers = {
        WorkerSpecialty.MATH: MathWorker,
        WorkerSpecialty.LOGIC: LogicWorker,
        WorkerSpecialty.CODE: CodeWorker,
        WorkerSpecialty.FACTUAL: FactualWorker,
        WorkerSpecialty.CREATIVE: CreativeWorker,
        WorkerSpecialty.ANALYSIS: AnalysisWorker,
    }
    
    worker_class = workers.get(specialty)
    if not worker_class:
        raise ValueError(f"No worker available for specialty: {specialty}")
    
    return worker_class(config, tree_cache)
