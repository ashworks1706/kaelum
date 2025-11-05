import time
import re
from typing import Dict, Any, Optional, List

from core.config import KaelumConfig
from core.tree_cache import TreeCache
from core.workers import WorkerAgent, WorkerResult, WorkerSpecialty
from core.reasoning import Message
from core.lats import LATS, LATSNode
from core.reward_model import RewardModel
from sentence_transformers import SentenceTransformer, util


class FactualWorker(WorkerAgent):
    def __init__(self, config: Optional[KaelumConfig] = None, tree_cache: Optional[TreeCache] = None):
        super().__init__(config, tree_cache)
        self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.FACTUAL
    
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
        
        query_type = self._classify_factual_query(query)
        
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
            
            completion = 0.0
            if "answer" in state:
                answer = state["answer"]
                completion = 1.0 if answer and len(answer) > 50 else 0.3
            else:
                facts_count = len(state.get("facts_gathered", []))
                completion = min(0.8, facts_count * 0.15)
            
            return RewardModel.get_reward("factual", state, depth, completion)
        
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
                prompt += "\n\nProvide additional relevant information or conclude."
            
            try:
                messages = [
                    Message(role="system", content=self.get_system_prompt()),
                    Message(role="user", content=prompt)
                ]
                response = self.llm_client.generate(messages)
                next_step = response.strip()
                
                has_conclusion = any(next_step.lower().startswith(kw) for kw in ['therefore', 'in conclusion', 'to summarize'])
                is_final = depth >= max_tree_depth - 1 or has_conclusion or len(next_step) > 200
                
                return {
                    "query": query,
                    "step": next_step,
                    "depth": depth + 1,
                    "query_type": query_type,
                    "facts_gathered": parent_state.get("facts_gathered", []) + ([next_step] if not is_final else []),
                    "answer": next_step if is_final else None
                }
            except:
                return {
                    "query": query,
                    "step": f"Gathering fact {depth + 1}",
                    "depth": depth + 1,
                    "query_type": query_type,
                    "facts_gathered": parent_state.get("facts_gathered", [])
                }
        
        tree = LATS(root_state, simulator=simulate_factual_step, expand_fn=expand_factual_step)
        
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
        
        answer = best_node.state.get("answer", reasoning_steps[-1] if reasoning_steps else "")
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
                'query_type': query_type,
                'num_simulations': num_simulations,
                'tree_depth': best_node.state.get("depth", 0),
                'cache_hit': False
            }
        )
    
    def _classify_factual_query(self, query: str) -> str:
        if not hasattr(self, '_type_embeddings'):
            type_exemplars = {
                'definition': "What is photosynthesis? Define machine learning.",
                'historical': "When did the Renaissance begin? What year did World War II end?",
                'geographical': "Where is Mount Everest located? What is the capital of Japan?",
                'quantitative': "How many planets are in our solar system? What is the population of China?",
                'biographical': "Who invented the telephone? Tell me about Marie Curie.",
                'general': "Explain the water cycle. Describe how airplanes fly."
            }
            
            self._type_names = list(type_exemplars.keys())
            self._type_embeddings = self._encoder.encode(
                list(type_exemplars.values()), 
                convert_to_tensor=True
            )
        
        query_embedding = self._encoder.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self._type_embeddings)[0]
        max_idx = int(similarities.argmax())
        
        return self._type_names[max_idx]
    
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
