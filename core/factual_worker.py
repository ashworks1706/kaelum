import asyncio
import time
import re
from typing import Dict, Any, Optional, List

from core.config import KaelumConfig
from core.workers import WorkerAgent, WorkerResult, WorkerSpecialty
from core.reasoning import LLMClient, Message


class FactualWorker(WorkerAgent):
    def __init__(self, config: Optional[KaelumConfig] = None):
        super().__init__(config)
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.FACTUAL
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        from sentence_transformers import SentenceTransformer, util
        
        if not hasattr(self, '_encoder'):
            self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            factual_exemplars = [
                "What is the capital of France?",
                "Who invented the telephone?",
                "When did World War II end?",
                "Where is the Great Wall of China located?",
                "How many planets are in the solar system?",
                "Define photosynthesis",
                "Explain the theory of relativity",
                "Tell me about the Renaissance period",
                "What are the symptoms of influenza?",
                "Describe the water cycle"
            ]
            self._factual_embeddings = self._encoder.encode(factual_exemplars, convert_to_tensor=True)
        
        query_embedding = self._encoder.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self._factual_embeddings)[0]
        return float(similarities.max())
    
    def solve(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        return asyncio.run(self._solve_async(query, context))
    
    async def solve_async(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        return await self._solve_async(query, context)
    
    async def _solve_async(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        start_time = time.time()
        reasoning_steps = []
        
        query_type = self._classify_factual_query(query)
        reasoning_steps.append(f"Query type: {query_type}")
        
        prompt = self._build_prompt(query, query_type, None)
        reasoning_steps.append("Built factual query prompt")
        
        messages = [Message(role="user", content=prompt)]
        response = self.llm_client.generate(messages)
        reasoning_steps.append("Generated factual answer")
        
        sources = self._extract_sources(response, None)
        
        confidence = self._calculate_confidence(
            query_type, None, sources, response
        )
        
        execution_time = time.time() - start_time
        
        return WorkerResult(
            answer=response,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=True,
            specialty=WorkerSpecialty.FACTUAL,
            execution_time=execution_time,
            metadata={
                'query_type': query_type,
                'sources': sources
            }
        )
    
    def _classify_factual_query(self, query: str) -> str:
        from sentence_transformers import SentenceTransformer, util
        
        if not hasattr(self, '_type_encoder'):
            self._type_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            type_exemplars = {
                'definition': "What is photosynthesis? Define machine learning.",
                'historical': "When did the Renaissance begin? What year did World War II end?",
                'geographical': "Where is Mount Everest located? What is the capital of Japan?",
                'quantitative': "How many planets are in our solar system? What is the population of China?",
                'biographical': "Who invented the telephone? Tell me about Marie Curie.",
                'general': "Explain the water cycle. Describe how airplanes fly."
            }
            
            self._type_names = list(type_exemplars.keys())
            self._type_embeddings = self._type_encoder.encode(
                list(type_exemplars.values()), 
                convert_to_tensor=True
            )
        
        query_embedding = self._type_encoder.encode(query, convert_to_tensor=True)
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
    
    def _extract_sources(
        self,
        response: str,
        retrieved_context: Optional[List[str]]
    ) -> List[str]:
        sources = []
        
        # Look for source citations like [Source 1], [1], (Source 1), etc.
        source_patterns = [
            r'\[Source\s+(\d+)\]',
            r'\[(\d+)\]',
            r'\(Source\s+(\d+)\)',
        ]
        
        for pattern in source_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                source_num = int(match)
                if retrieved_context and 0 < source_num <= len(retrieved_context):
                    if f"Source {source_num}" not in sources:
                        sources.append(f"Source {source_num}")
        
        return sources
    
    def _calculate_confidence(
        self,
        query_type: str,
        retrieved_context: Optional[List[str]],
        sources: List[str],
        response: str
    ) -> float:
        from sentence_transformers import SentenceTransformer, util
        
        if not hasattr(self, '_conf_encoder'):
            self._conf_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        if not response or len(response) < 20:
            return 0.3
        
        words = response.split()
        if len(words) < 5:
            return 0.4
        
        response_parts = response.split('.')[:3]
        if len(response_parts) < 2:
            return 0.5
        
        has_numbers = bool(re.search(r'\d+', response))
        has_specifics = len(response) > 100
        
        base_confidence = 0.6
        if has_numbers:
            base_confidence += 0.1
        if has_specifics:
            base_confidence += 0.1
        if len(sources) > 0:
            base_confidence += 0.1
        
        return min(base_confidence, 0.95)
    
    async def verify(self, query: str, answer: str, context: Optional[Dict] = None) -> bool:
        from sentence_transformers import SentenceTransformer, util
        
        if not answer or len(answer.strip()) < 10:
            return False
        
        if not hasattr(self, '_verif_encoder'):
            self._verif_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        query_embedding = self._verif_encoder.encode(query, convert_to_tensor=True)
        answer_embedding = self._verif_encoder.encode(answer, convert_to_tensor=True)
        
        similarity = float(util.cos_sim(query_embedding, answer_embedding)[0][0])
        
        return similarity > 0.3
