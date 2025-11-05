import asyncio
import time
import re
from typing import Dict, Any, Optional, List

from core.config import KaelumConfig
from core.tree_cache import TreeCache
from core.workers import WorkerAgent, WorkerResult, WorkerSpecialty
from core.reasoning import LLMClient, Message


class CreativeWorker(WorkerAgent):
    def __init__(self, config: Optional[KaelumConfig] = None, tree_cache: Optional[TreeCache] = None):
        super().__init__(config, tree_cache)
        base_temp = self.config.reasoning_llm.temperature
        self.creative_temperature = min(base_temp + 0.3, 1.0)
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.CREATIVE
    
    def get_system_prompt(self) -> str:
        return """You are a creative writing and ideation expert specializing in:
        - Story and narrative creation
        - Poetry and prose composition
        - Creative brainstorming
        - Content generation (articles, blogs, essays)
        - Character and dialogue development
        - Innovative problem-solving
        - Conceptual design and ideation

        Provide imaginative, original, and engaging creative content.
        Use vivid language and explore multiple perspectives."""
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        return 1.0
    
    def solve(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        return asyncio.run(self._solve_async(query, context))
    
    async def solve_async(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        return await self._solve_async(query, context)
    
    async def _solve_async(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        start_time = time.time()
        reasoning_steps = []
        
        # Determine creative task type
        task_type = self._classify_creative_task(query)
        reasoning_steps.append(f"Creative task type: {task_type}")
        
        # Build creative prompt
        prompt = self._build_creative_prompt(query, task_type)
        reasoning_steps.append("Built creative prompt with enhanced temperature")
        
        # Generate creative response
        # Note: We'd ideally pass temperature here, but LLMClient.generate() uses config.temperature
        # TODO: Consider enhancing LLMClient to accept optional temperature override
        messages = [Message(role="user", content=prompt)]
        response = self.llm_client.generate(messages)
        reasoning_steps.append("Generated creative response")
        
        # Analyze creativity metrics
        metrics = self._analyze_creativity(response, task_type)
        reasoning_steps.append(f"Creativity metrics: diversity={metrics['diversity']:.2f}, coherence={metrics['coherence']:.2f}")
        
        # Calculate confidence based on coherence and completeness
        confidence = self._calculate_confidence(response, task_type, metrics)
        
        execution_time = time.time() - start_time
        
        return WorkerResult(
            answer=response,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=metrics['coherence'] > 0.6,
            specialty=WorkerSpecialty.CREATIVE,
            execution_time=execution_time,
            metadata={
                'task_type': task_type,
                'diversity_score': metrics['diversity'],
                'coherence_score': metrics['coherence'],
                'temperature': self.creative_temperature,
                'word_count': len(response.split())
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
    
    async def verify(self, query: str, answer: str, context: Optional[Dict] = None) -> bool:
        # Basic checks
        if not answer or len(answer.strip()) < 20:
            return False
        
        # Check for minimum content
        words = answer.split()
        if len(words) < 10:
            return False
        
        # Check for some structure (sentences or lines)
        has_structure = '.' in answer or '\n' in answer or '!' in answer or '?' in answer
        if not has_structure:
            return False
        
        return True
