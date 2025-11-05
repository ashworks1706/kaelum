import time
from typing import Dict, Any, Optional, List

from core.config import KaelumConfig
from core.tree_cache import TreeCache
from core.workers import WorkerAgent, WorkerResult, WorkerSpecialty
from core.reasoning import Message
from core.lats import LATS, LATSNode
from core.reward_model import RewardModel
from core.conclusion_detector import ConclusionDetector
from core.adaptive_penalty import AdaptivePenalty


class AnalysisWorker(WorkerAgent):
    def __init__(self, config: Optional[KaelumConfig] = None, tree_cache: Optional[TreeCache] = None):
        super().__init__(config, tree_cache)
        self.conclusion_detector = ConclusionDetector()
        self.relevance_validator = RelevanceValidator()
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.ANALYSIS
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        return 1.0
    
    def solve(self, query: str, context: Optional[Dict] = None,
              use_cache: bool = True, max_tree_depth: int = 5,
              num_simulations: int = 10, parallel: bool = False) -> WorkerResult:
        start_time = time.time()
        
        if use_cache:
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
                prompt += "\n\nWhat is the next analytical insight or conclusion?"
            else:
                prompt += "\n\nWhat is the first analytical observation?"
            
            try:
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
            except:
                return {
                    "query": query,
                    "step": f"Analysis point {depth + 1}",
                    "depth": depth + 1,
                    "analysis_points": parent_state.get("analysis_points", [])
                }
        
        tree = LATS(root_state, simulator=simulate_analysis_step, expand_fn=expand_analysis_step)
        
        tree.run_simulations(num_simulations, max_tree_depth, parallel=parallel)
        
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
        
        return WorkerResult(
            answer=answer,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=False,
            specialty=self.get_specialty(),
            execution_time=execution_time,
            metadata={
                'num_simulations': num_simulations,
                'tree_depth': best_node.state.get("depth", 0),
                'cache_hit': False
            }
        )
