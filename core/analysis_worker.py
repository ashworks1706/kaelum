import time
from typing import Dict, Any, Optional, List

from core.config import KaelumConfig
from core.tree_cache import TreeCache
from core.workers import WorkerAgent, WorkerResult, WorkerSpecialty
from core.reasoning import Message
from core.lats import LATS, LATSNode


class AnalysisWorker(WorkerAgent):
    def __init__(self, config: Optional[KaelumConfig] = None, tree_cache: Optional[TreeCache] = None):
        super().__init__(config, tree_cache)
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.ANALYSIS
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        return 1.0
    
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
            "step": "Initiating analytical reasoning",
            "depth": 0,
            "analysis_points": []
        }
        
        def simulate_analysis_step(node: LATSNode) -> float:
            state = node.state
            
            if "conclusion" in state:
                conclusion = state["conclusion"]
                if conclusion and len(conclusion) > 50:
                    return 0.9
                return 0.4
            
            points_count = len(state.get("analysis_points", []))
            if points_count > 0:
                return 0.6 + (points_count * 0.05)
            
            depth = state.get("depth", 0)
            return max(0.2, 0.5 - depth * 0.05)
        
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
                
                has_conclusion = any(next_step.lower().startswith(kw) for kw in ['therefore', 'in conclusion', 'overall', 'to summarize'])
                is_final = depth >= max_tree_depth - 1 or has_conclusion or len(next_step) > 200
                
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
        
        for _ in range(num_simulations):
            node = tree.select()
            if node.state.get("depth", 0) >= max_tree_depth:
                continue
            child_state = expand_analysis_step(node)
            child = tree.expand(node, child_state)
            reward = simulate_analysis_step(child)
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
        verification_passed = confidence > 0.7 and len(answer) > 30
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
                'num_simulations': num_simulations,
                'tree_depth': best_node.state.get("depth", 0),
                'cache_hit': False
            }
        )
