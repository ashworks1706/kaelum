import time
import re
from typing import Dict, Any, Optional, List

from core.config import KaelumConfig
from core.tree_cache import TreeCache
from core.workers import WorkerAgent, WorkerResult, WorkerSpecialty
from core.reasoning import Message
from core.lats import LATS, LATSNode
from core.reward_model import RewardModel
from core.adaptive_penalty import AdaptivePenalty


class CreativeWorker(WorkerAgent):
    def __init__(self, config: Optional[KaelumConfig] = None, tree_cache: Optional[TreeCache] = None):
        super().__init__(config, tree_cache)
        base_temp = self.config.reasoning_llm.temperature
        self.creative_temperature = min(base_temp + 0.3, 1.0)
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.CREATIVE
    
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
        
        task_type = self._classify_creative_task(query)
        
        root_state = {
            "query": query,
            "step": f"Initiating {task_type} creation",
            "depth": 0,
            "task_type": task_type,
            "content_parts": []
        }
        
        def simulate_creative_step(node: LATSNode) -> float:
            state = node.state
            depth = state.get("depth", 0)
            
            has_content = state.get("content") is not None
            has_parts = len(state.get("content_parts", [])) > 0
            
            query_complexity = AdaptivePenalty.compute_complexity(query)
            
            return RewardModel.get_reward("creative", state, depth, has_content, has_parts,
                                         query_complexity=query_complexity)
        
        def expand_creative_step(parent_node: LATSNode) -> Dict[str, Any]:
            parent_state = parent_node.state
            depth = parent_state.get("depth", 0)
            
            history = []
            node = parent_node
            while node.parent is not None:
                if "step" in node.state:
                    history.insert(0, node.state["step"])
                node = node.parent
            
            prompt = self._build_creative_prompt(query, task_type)
            if history:
                prompt += "\n\nContent created so far:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(history))
                prompt += "\n\nContinue or conclude the creative work."
            
            try:
                messages = [
                    Message(role="system", content=self.get_system_prompt()),
                    Message(role="user", content=prompt)
                ]
                response = self.llm_client.generate(messages)
                next_step = response.strip()
                
                is_substantial = len(next_step) > 150
                is_final = depth >= max_tree_depth - 1 or is_substantial
                
                return {
                    "query": query,
                    "step": next_step,
                    "depth": depth + 1,
                    "task_type": task_type,
                    "content_parts": parent_state.get("content_parts", []) + ([next_step] if not is_final else []),
                    "content": next_step if is_final else None
                }
            except:
                return {
                    "query": query,
                    "step": f"Creative iteration {depth + 1}",
                    "depth": depth + 1,
                    "task_type": task_type,
                    "content_parts": parent_state.get("content_parts", [])
                }
        
        tree = LATS(root_state, simulator=simulate_creative_step, expand_fn=expand_creative_step)
        
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
        
        answer = best_node.state.get("content", reasoning_steps[-1] if reasoning_steps else "")
        metrics = self._analyze_creativity(answer, task_type)
        confidence = RewardModel.compute_confidence(best_node.value if best_node else 0.0, best_node.visits if best_node else 0, tree.root.visits)
        verification_passed = metrics['coherence'] > 0.6 and len(answer) > 50
        execution_time = time.time() - start_time
        
        if use_cache:
            self.tree_cache.store(query, tree, self.get_specialty().value,
                                 verification_passed, confidence)
        
        return WorkerResult(
            answer=answer,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=False,
            specialty=self.get_specialty(),
            execution_time=execution_time,
            metadata={
                'task_type': task_type,
                'diversity_score': metrics['diversity'],
                'coherence_score': metrics['coherence'],
                'num_simulations': num_simulations,
                'tree_depth': best_node.state.get("depth", 0),
                'cache_hit': False
            }
        )
    
    def _classify_creative_task(self, query: str) -> str:
        
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['story', 'narrative', 'fiction', 'tale']):
            return 'storytelling'
        elif any(kw in query_lower for kw in ['poem', 'verse', 'rhyme', 'haiku']):
            return 'poetry'
        elif any(kw in query_lower for kw in ['essay', 'article', 'blog', 'post']):
            return 'writing'
        elif any(kw in query_lower for kw in ['brainstorm', 'idea', 'suggest', 'propose']):
            return 'ideation'
        elif any(kw in query_lower for kw in ['design', 'concept', 'plan']):
            return 'design'
        elif any(kw in query_lower for kw in ['dialogue', 'conversation', 'script']):
            return 'dialogue'
        else:
            return 'general_creative'
    
    def _build_creative_prompt(self, query: str, task_type: str) -> str:
        prompt_parts = []
        
        # Base instruction emphasizing creativity
        prompt_parts.append("You are a creative expert. Think imaginatively and generate novel, engaging content.")
        
        # Task-type specific instructions
        if task_type == 'storytelling':
            prompt_parts.append("\nFocus on narrative structure, character development, and engaging plot.")
        elif task_type == 'poetry':
            prompt_parts.append("\nFocus on imagery, rhythm, and emotional resonance.")
        elif task_type == 'writing':
            prompt_parts.append("\nFocus on clarity, structure, and compelling arguments or insights.")
        elif task_type == 'ideation':
            prompt_parts.append("\nGenerate diverse, innovative ideas. Think outside the box.")
        elif task_type == 'design':
            prompt_parts.append("\nConsider aesthetics, functionality, and user experience.")
        elif task_type == 'dialogue':
            prompt_parts.append("\nCreate natural, engaging dialogue with distinct voices.")
        
        # Encourage exploration
        prompt_parts.append("\nBe creative, original, and don't be afraid to take risks.")
        
        # Add the query
        prompt_parts.append(f"\n\nTask: {query}")
        prompt_parts.append("\nResponse:")
        
        return "\n".join(prompt_parts)
    
    def _analyze_creativity(self, response: str, task_type: str) -> Dict[str, float]:
        metrics = {
            'diversity': 0.0,
            'coherence': 0.0
        }
        
        # Measure diversity via vocabulary richness
        words = response.lower().split()
        if words:
            unique_words = set(words)
            metrics['diversity'] = min(len(unique_words) / len(words), 1.0)
        
        # Measure coherence via basic heuristics
        sentences = response.split('.')
        
        # Check for reasonable sentence length
        if sentences:
            avg_sentence_length = len(words) / len(sentences)
            if 5 <= avg_sentence_length <= 30:
                metrics['coherence'] += 0.3
        
        # Check for completeness (reasonable length)
        if len(words) >= 20:
            metrics['coherence'] += 0.3
        
        # Check for structure (paragraphs or line breaks)
        if '\n' in response or len(response) > 100:
            metrics['coherence'] += 0.2
        
        # Task-specific bonuses
        if task_type == 'poetry' and len(response.split('\n')) > 2:
            metrics['coherence'] += 0.2  # Multiple lines
        
        metrics['coherence'] = min(metrics['coherence'], 1.0)
        
        return metrics
    
    def _calculate_confidence(
        self,
        response: str,
        task_type: str,
        metrics: Dict[str, float]
    ) -> float:
        confidence = 0.4  # Base confidence for creative tasks
        
        # Bonus for good coherence
        confidence += metrics['coherence'] * 0.3
        
        # Bonus for good diversity
        confidence += metrics['diversity'] * 0.2
        
        # Bonus for adequate length
        word_count = len(response.split())
        if word_count >= 50:
            confidence += 0.1
        elif word_count < 20:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
