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
        self.factual_keywords = [
            'what is', 'who is', 'when did', 'where is', 'how many',
            'define', 'explain', 'describe', 'tell me about',
            'fact', 'information', 'data', 'statistics', 'history',
            'capital', 'population', 'year', 'date', 'location',
            'definition', 'meaning', 'called', 'known as'
        ]
    
    def get_specialty(self) -> WorkerSpecialty:
        return WorkerSpecialty.FACTUAL
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        query_lower = query.lower()
        score = 0.0
        
        # Check for factual question patterns (higher weight)
        question_patterns = [
            r'\bwhat\s+is\b',
            r'\bwho\s+is\b',
            r'\bwhen\s+did\b',
            r'\bwhere\s+is\b',
            r'\bhow\s+many\b',
            r'\bwhich\s+\w+\b',
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, query_lower):
                score += 0.5
                break
        
        # Check for factual keywords
        keyword_count = sum(1 for kw in self.factual_keywords if kw in query_lower)
        score += min(keyword_count * 0.2, 0.4)
        
        # Check for specific factual indicators
        if any(ind in query_lower for ind in ['capital', 'population', 'year', 'date']):
            score += 0.25
        
        # Check for definition requests (higher weight)
        if 'define' in query_lower or 'definition' in query_lower or 'meaning' in query_lower:
            score += 0.4
        
        # Penalize if it looks like other types
        if any(word in query_lower for word in ['solve', 'calculate', 'prove', 'write code']):
            score *= 0.5
        
        return min(score, 1.0)
    
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
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['define', 'definition', 'meaning', 'what is', 'what are']):
            return 'definition'
        elif any(kw in query_lower for kw in ['history', 'when', 'year', 'date']):
            return 'historical'
        elif any(kw in query_lower for kw in ['where', 'location', 'place']):
            return 'geographical'
        elif any(kw in query_lower for kw in ['how many', 'population', 'number', 'statistics']):
            return 'quantitative'
        elif any(kw in query_lower for kw in ['who', 'person', 'people']):
            return 'biographical'
        else:
            return 'general'
    
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
        confidence = 0.5
        
        if query_type in ['definition', 'biographical']:
            confidence += 0.2
        elif query_type == 'quantitative':
            if re.search(r'\d+', response):
                confidence += 0.15
        
        if len(response) < 50:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    async def verify(self, query: str, answer: str, context: Optional[Dict] = None) -> bool:
        if not answer or len(answer.strip()) < 10:
            return False
        
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'who', 'when', 'where', 'how'}
        query_keywords = query_words - common_words
        
        overlap = len(query_keywords & answer_words)
        if overlap == 0 and len(query_keywords) > 0:
            return False
        
        return True
