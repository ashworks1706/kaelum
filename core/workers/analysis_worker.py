import time
import logging
from typing import Dict, Any, Optional, List

from ..config import KaelumConfig
from ..search import TreeCache
from .workers import WorkerAgent, WorkerResult, WorkerSpecialty
from ..reasoning import Message
from ..search import LATS, LATSNode
from ..search import RewardModel
from ..detectors import ConclusionDetector
from ..learning import AdaptivePenalty
from ..verification import RelevanceValidator

logger = logging.getLogger("kaelum.analysis_worker")

class AnalysisWorker(WorkerAgent):
    def __init__(self, config: Optional[KaelumConfig] = None, tree_cache: Optional[TreeCache] = None):
        super().__init__(config, tree_cache)
        self.conclusion_detector = ConclusionDetector(embedding_model=self.config.embedding_model)
        self.relevance_validator = RelevanceValidator(embedding_model=self.config.embedding_model)
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.ANALYSIS
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        return 1.0
    
    def solve(self, query: str, context: Optional[Dict] = None,
              use_cache: bool = True, max_tree_depth: int = 5,
              num_simulations: int = 10, parallel: bool = False,
              max_workers: int = 4,
              existing_tree=None,
              extra_sims: int = 0,
              verification_issues=None) -> WorkerResult:
        start_time = time.time()
        
        if use_cache and existing_tree is None:
            cached_result = self._check_cache(query)
            if cached_result:
                return cached_result
        
        root_state = {
            "query": query,
            "step": "Initiating analytical reasoning",
            "depth": 0,
            "analysis_points": []
        }
        
        def simulate_analysis_step(node: LATSNode) -> float:
            state = node.state
            depth = state.get("depth", 0)
            
            has_conclusion = state.get("conclusion") is not None
            has_points = len(state.get("analysis_points", [])) > 0
            
            query_complexity = AdaptivePenalty.compute_complexity(query)
            
            return RewardModel.get_reward("analysis", state, depth, has_conclusion, has_points,
                                         query_complexity=query_complexity)
        
        def expand_analysis_step(parent_node: LATSNode) -> Dict[str, Any]:
            parent_state = parent_node.state
            depth = parent_state.get("depth", 0)
            
            history = []
            node = parent_node
            while node.parent is not None:
                if "step" in node.state:
                    history.insert(0, node.state["step"])
                node = node.parent
            
            prompt = f"Query: {query}\n\nAnalyze this systematically and critically."
            if history:
                prompt += "\n\nAnalysis so far:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(history))
                prompt += "\n\nProvide ONLY the next analytical insight or observation. Keep it concise (1-3 sentences). Do not provide the complete analysis yet."
            else:
                prompt += "\n\nProvide ONLY the first analytical observation. Keep it concise (1-3 sentences)."
            
            messages = [
                Message(role="system", content=self.get_system_prompt()),
                Message(role="user", content=prompt)
            ]
            response = self.llm_client.generate(messages)
            next_step = response.strip()
            
            conclusion_result = self.conclusion_detector.detect(next_step, history)
            is_final = depth >= max_tree_depth - 1 or conclusion_result['is_conclusion'] or len(next_step) > 200
            
            return {
                "query": query,
                "step": next_step,
                "depth": depth + 1,
                "analysis_points": parent_state.get("analysis_points", []) + ([next_step] if not is_final else []),
                "conclusion": next_step if is_final else None
            }
        if existing_tree is not None:
            tree = existing_tree
            tree.simulator = simulate_analysis_step
            tree.expand_fn = expand_analysis_step
            tree.coherence_checker = self._lightweight_coherence_check
            self._penalize_failed_path(tree, verification_issues or [])
            sims = extra_sims if extra_sims > 0 else max(3, num_simulations // 2)
            logger.info(f"TREE-REUSE: Continuing analysis search ({sims} additional simulations)")
        else:
            tree = LATS(root_state, simulator=simulate_analysis_step, expand_fn=expand_analysis_step)
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
        confidence = RewardModel.compute_confidence(best_node.value if best_node else 0.0, best_node.visits if best_node else 0, tree.root.visits)
        
        execution_time = time.time() - start_time
        
        if use_cache:
            self.tree_cache.store(query, tree, self.get_specialty().value,
                                 False, confidence)
        
        avg_reward = tree.get_avg_reward()
        
        return WorkerResult(
            answer=answer,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=False,
            specialty=self.get_specialty(),
            execution_time=execution_time,
            lats_tree=tree,
            metadata={
                'num_simulations': num_simulations,
                'tree_depth': best_node.state.get("depth", 0),
                'cache_hit': False,
                'avg_reward': avg_reward
            }
        )
