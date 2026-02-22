import time
import logging
import re
from typing import Dict, Any, Optional, List

from ..config import KaelumConfig
from ..search import TreeCache
from .workers import WorkerAgent, WorkerResult, WorkerSpecialty
from ..reasoning import Message
from ..search import LATS, LATSNode
from ..search import RewardModel

logger = logging.getLogger("kaelum.factual_worker")
from ..learning import AdaptivePenalty
from sentence_transformers import SentenceTransformer, util

class FactualWorker(WorkerAgent):
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', config: Optional[KaelumConfig] = None, tree_cache: Optional[TreeCache] = None):
        super().__init__(config, tree_cache)
        self._encoder = SentenceTransformer(embedding_model)
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.FACTUAL
    
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
        
        query_type = "general"
        
        root_state = {
            "query": query,
            "step": f"Analyzing {query_type} query",
            "depth": 0,
            "query_type": query_type,
            "facts_gathered": []
        }
        
        def simulate_factual_step(node: LATSNode) -> float:
            state = node.state
            depth = state.get("depth", 0)
            
            has_answer = state.get("answer") is not None
            has_facts = len(state.get("facts_gathered", [])) > 0
            
            query_complexity = AdaptivePenalty.compute_complexity(query)
            
            return RewardModel.get_reward("factual", state, depth, has_answer, has_facts,
                                         query_complexity=query_complexity)
        
        def expand_factual_step(parent_node: LATSNode) -> Dict[str, Any]:
            parent_state = parent_node.state
            depth = parent_state.get("depth", 0)
            
            history = []
            node = parent_node
            while node.parent is not None:
                if "step" in node.state:
                    history.insert(0, node.state["step"])
                node = node.parent
            
            prompt = self._build_prompt(query, query_type, None)
            if history:
                prompt += "\n\nFacts gathered so far:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(history))
                prompt += "\n\nProvide ONLY the next relevant fact or information piece. Keep it concise (1-3 sentences). Do not provide the complete answer yet."
            
            messages = [
                Message(role="system", content=self.get_system_prompt()),
                Message(role="user", content=prompt)
            ]
            response = self.llm_client.generate(messages)
            next_step = response.strip()
            
            history = []
            node = parent_node
            while node.parent is not None:
                if "step" in node.state:
                    history.insert(0, node.state["step"])
                node = node.parent
            
            conclusion_result = self.conclusion_detector.detect(next_step, '\n'.join(history))
            has_conclusion = conclusion_result['is_conclusion'] and conclusion_result['confidence'] > 0.7
            
            is_final = depth >= max_tree_depth - 1
            
            return {
                "query": query,
                "step": next_step,
                "depth": depth + 1,
                "query_type": query_type,
                "facts_gathered": parent_state.get("facts_gathered", []) + ([next_step] if not is_final else []),
                "answer": next_step if is_final else None
            }
        if existing_tree is not None:
            tree = existing_tree
            tree.simulator = simulate_factual_step
            tree.expand_fn = expand_factual_step
            tree.coherence_checker = None
            self._penalize_failed_path(tree, verification_issues or [])
            sims = extra_sims if extra_sims > 0 else max(3, num_simulations // 2)
            logger.info(f"TREE-REUSE: Continuing factual search ({sims} additional simulations)")
        else:
            tree = self._build_lats(root_state, simulate_factual_step, expand_factual_step)
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
        
        answer = best_node.state.get("answer", reasoning_steps[-1] if reasoning_steps else "")
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
                'query_type': query_type,
                'num_simulations': num_simulations,
                'tree_depth': best_node.state.get("depth", 0),
                'cache_hit': False,
                'avg_reward': avg_reward
            }
        )
    
    def _build_prompt(
        self,
        query: str,
        query_type: str,
        retrieved_context: Optional[List[str]]
    ) -> str:
        prompt_parts = []
        
        prompt_parts.append("You are a factual information expert. Provide accurate, well-sourced answers.")
        
        if query_type == 'definition':
            prompt_parts.append("\nProvide a clear, concise definition with examples if helpful.")
        elif query_type == 'historical':
            prompt_parts.append("\nProvide accurate dates and historical context.")
        elif query_type == 'geographical':
            prompt_parts.append("\nProvide specific location information and relevant geographical details.")
        elif query_type == 'quantitative':
            prompt_parts.append("\nProvide specific numbers, statistics, and their sources.")
        elif query_type == 'biographical':
            prompt_parts.append("\nProvide accurate information about the person including key achievements.")
        
        prompt_parts.append(f"\n\nQuestion: {query}")
        prompt_parts.append("\nAnswer:")
        
        return "\n".join(prompt_parts)
