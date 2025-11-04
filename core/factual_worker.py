"""FactualWorker - Specialized worker for factual queries.

This worker handles factual queries by:
1. Analyzing the question to identify what facts are needed
2. Reasoning through available knowledge
3. Providing accurate, well-reasoned answers
"""

import asyncio
import time
import re
from typing import Dict, Any, Optional, List

from core.config import KaelumConfig
from core.workers import WorkerAgent, WorkerResult, WorkerSpecialty
from core.reasoning import LLMClient, Message


class FactualWorker(WorkerAgent):
    """Worker specialized in factual queries."""
    
    def __init__(self, config: Optional[KaelumConfig] = None):
        """Initialize FactualWorker.
        
        Args:
            config: Kaelum configuration
        """
        super().__init__(config)
        self.factual_keywords = [
            'what is', 'who is', 'when did', 'where is', 'how many',
            'define', 'explain', 'describe', 'tell me about',
            'fact', 'information', 'data', 'statistics', 'history',
            'capital', 'population', 'year', 'date', 'location',
            'definition', 'meaning', 'called', 'known as'
        ]
    
    def get_specialty(self) -> WorkerSpecialty:
        """Return worker specialty."""
        return WorkerSpecialty.FACTUAL
    
    def can_handle(self, query: str, context: Optional[Dict] = None) -> float:
        """Determine if this worker can handle the query.
        
        Args:
            query: The query to evaluate
            context: Optional context
            
        Returns:
            Confidence score between 0 and 1
        """
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
        """Synchronous solve method (calls async version).
        
        Args:
            query: The query to solve
            context: Optional context
            
        Returns:
            WorkerResult with factual answer
        """
        return asyncio.run(self._solve_async(query, context))
    
    async def solve_async(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        """Async solve method for parallel execution."""
        return await self._solve_async(query, context)
    
    async def _solve_async(self, query: str, context: Optional[Dict] = None) -> WorkerResult:
        """Solve a factual query.
        
        Args:
            query: The query to solve
            context: Optional context
            
        Returns:
            WorkerResult with factual answer
        """
        start_time = time.time()
        reasoning_steps = []
        
        # Detect query type
        query_type = self._classify_factual_query(query)
        reasoning_steps.append(f"Query type: {query_type}")
        
        # Retrieve relevant context if RAG adapter available
        retrieved_context = None
        if self.rag_adapter:
            try:
                retrieved_context = await self.rag_adapter.retrieve(query, top_k=5)
                reasoning_steps.append(f"Retrieved {len(retrieved_context)} relevant documents")
            except Exception as e:
                reasoning_steps.append(f"RAG retrieval failed: {str(e)}")
        
        # Build prompt with retrieved context
        prompt = self._build_prompt(query, query_type, retrieved_context)
        reasoning_steps.append("Built factual query prompt")
        
        # Generate answer
        messages = [Message(role="user", content=prompt)]
        response = self.llm_client.generate(messages)
        reasoning_steps.append("Generated factual answer")
        
        # Extract sources if present
        sources = self._extract_sources(response, retrieved_context)
        
        # Calculate confidence based on source availability and query type
        confidence = self._calculate_confidence(
            query_type, retrieved_context, sources, response
        )
        
        execution_time = time.time() - start_time
        
        return WorkerResult(
            answer=response,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            verification_passed=retrieved_context is not None,
            specialty=WorkerSpecialty.FACTUAL,
            execution_time=execution_time,
            metadata={
                'query_type': query_type,
                'has_rag': self.rag_adapter is not None,
                'retrieved_docs': len(retrieved_context) if retrieved_context else 0,
                'sources': sources
            }
        )
    
    def _classify_factual_query(self, query: str) -> str:
        """Classify the type of factual query.
        
        Args:
            query: The query text
            
        Returns:
            Query type string
        """
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
        """Build prompt for factual query.
        
        Args:
            query: Original query
            query_type: Type of factual query
            retrieved_context: Retrieved documents from RAG
            
        Returns:
            Formatted prompt
        """
        prompt_parts = []
        
        # Base instruction
        prompt_parts.append("You are a factual information expert. Provide accurate, well-sourced answers.")
        
        # Query-type specific instructions
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
        
        # Add retrieved context if available
        if retrieved_context:
            prompt_parts.append("\n\nRelevant Context:")
            for i, doc in enumerate(retrieved_context[:5], 1):
                prompt_parts.append(f"\n[Source {i}]: {doc}")
            prompt_parts.append("\n\nUse the above context to answer accurately. Cite sources when possible.")
        else:
            prompt_parts.append("\n\nNote: No external context available. Answer based on your knowledge.")
        
        # Add the query
        prompt_parts.append(f"\n\nQuestion: {query}")
        prompt_parts.append("\nAnswer:")
        
        return "\n".join(prompt_parts)
    
    def _extract_sources(
        self,
        response: str,
        retrieved_context: Optional[List[str]]
    ) -> List[str]:
        """Extract cited sources from response.
        
        Args:
            response: Generated response
            retrieved_context: Retrieved documents
            
        Returns:
            List of source citations
        """
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
        """Calculate confidence score.
        
        Args:
            query_type: Type of query
            retrieved_context: Retrieved documents
            sources: Extracted source citations
            response: Generated response
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence
        
        # Bonus for RAG retrieval
        if retrieved_context:
            confidence += 0.2
            
            # Additional bonus if sources were cited
            if sources:
                confidence += 0.15
        
        # Adjust for query type complexity
        if query_type in ['definition', 'biographical']:
            confidence += 0.1  # Straightforward queries
        elif query_type == 'quantitative':
            # Check if response contains numbers
            if re.search(r'\d+', response):
                confidence += 0.05
        
        # Penalize very short responses
        if len(response) < 50:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    async def verify(self, query: str, answer: str, context: Optional[Dict] = None) -> bool:
        """Verify factual answer.
        
        For factual queries, we check:
        1. Answer is not empty
        2. Answer is relevant to query
        3. Sources are cited if RAG available
        
        Args:
            query: Original query
            answer: Generated answer
            context: Optional context
            
        Returns:
            True if verification passes
        """
        # Basic checks
        if not answer or len(answer.strip()) < 10:
            return False
        
        # Check if answer contains relevant keywords from query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'who', 'when', 'where', 'how'}
        query_keywords = query_words - common_words
        
        # At least some query keywords should appear in answer
        overlap = len(query_keywords & answer_words)
        if overlap == 0 and len(query_keywords) > 0:
            return False
        
        # If RAG was used, prefer answers with citations
        if self.rag_adapter:
            has_citations = bool(re.search(r'\[Source\s+\d+\]|\[\d+\]', answer))
            if not has_citations:
                # Still valid but lower confidence
                pass
        
        return True
