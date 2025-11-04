"""Meta-Reasoning system for combining multiple worker results.

The MetaReasoner coordinates multiple workers and combines their outputs using
various strategies (voting, confidence-based, synthesis, verification).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any
from collections import Counter

from kaelum.core.workers import WorkerAgent, WorkerResult, WorkerSpecialty
from kaelum.core.reasoning import LLMClient, Message


class CombinationStrategy(Enum):
    """Strategies for combining multiple worker results."""
    VOTING = "voting"              # Majority vote on answers
    CONFIDENCE = "confidence"      # Pick highest confidence
    VERIFICATION = "verification"  # Pick most verified
    SYNTHESIS = "synthesis"        # LLM combines all results
    WEIGHTED = "weighted"          # Weighted by specialty + confidence


@dataclass
class MetaResult:
    """Result from meta-reasoning combining multiple workers.
    
    Attributes:
        answer: The final combined answer
        confidence: Combined confidence score
        strategy: Which combination strategy was used
        worker_results: Individual results from all workers
        reasoning: Explanation of how answer was derived
        execution_time: Total time including all workers
        metadata: Additional information about combination
    """
    answer: str
    confidence: float
    strategy: CombinationStrategy
    worker_results: List[WorkerResult]
    reasoning: str
    execution_time: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "strategy": self.strategy.value,
            "worker_results": [
                {
                    "specialty": r.specialty.value,
                    "answer": r.answer,
                    "confidence": r.confidence,
                    "verification_passed": r.verification_passed
                }
                for r in self.worker_results
            ],
            "reasoning": self.reasoning,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }


class MetaReasoner:
    """Coordinates multiple workers and combines their results.
    
    The MetaReasoner runs multiple workers in parallel, then combines their
    results using configurable strategies to produce a higher-quality answer.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize MetaReasoner.
        
        Args:
            llm_client: LLM client for synthesis strategy
        """
        self.llm_client = llm_client
        self.workers: List[WorkerAgent] = []
        
    def add_worker(self, worker: WorkerAgent):
        """Add a worker to the meta-reasoner.
        
        Args:
            worker: Worker to add
        """
        self.workers.append(worker)
        
    def add_workers(self, workers: List[WorkerAgent]):
        """Add multiple workers.
        
        Args:
            workers: List of workers to add
        """
        self.workers.extend(workers)
    
    async def reason_async(
        self,
        query: str,
        strategy: CombinationStrategy = CombinationStrategy.CONFIDENCE,
        context: Optional[Dict] = None,
        worker_filter: Optional[List[WorkerSpecialty]] = None
    ) -> MetaResult:
        """Run multiple workers and combine results asynchronously.
        
        Args:
            query: The query to solve
            strategy: How to combine worker results
            context: Optional context for workers
            worker_filter: Only use workers with these specialties
            
        Returns:
            MetaResult with combined answer
        """
        import time
        start_time = time.time()
        
        # Filter workers if requested
        if worker_filter:
            workers_to_use = [
                w for w in self.workers 
                if w.get_specialty() in worker_filter
            ]
        else:
            # Auto-select workers based on can_handle scores
            worker_scores = [
                (w, w.can_handle(query, context))
                for w in self.workers
            ]
            # Use workers with confidence > 0.3
            workers_to_use = [
                w for w, score in worker_scores if score > 0.3
            ]
        
        if not workers_to_use:
            # No workers scored high enough - use all for maximum coverage
            workers_to_use = self.workers
            
        # Run all workers in parallel
        tasks = [
            worker.solve_async(query, context)
            for worker in workers_to_use
        ]
        
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [
            r for r in worker_results 
            if isinstance(r, WorkerResult) and not r.error
        ]
        
        if not valid_results:
            # All workers failed
            return MetaResult(
                answer="Error: All workers failed to produce results",
                confidence=0.0,
                strategy=strategy,
                worker_results=[],
                reasoning="No valid worker results",
                execution_time=time.time() - start_time,
                metadata={"error": "all_workers_failed"}
            )
        
        # Combine results using selected strategy
        combined = self._combine_results(valid_results, strategy, query)
        
        execution_time = time.time() - start_time
        
        return MetaResult(
            answer=combined["answer"],
            confidence=combined["confidence"],
            strategy=strategy,
            worker_results=valid_results,
            reasoning=combined["reasoning"],
            execution_time=execution_time,
            metadata={
                "num_workers": len(valid_results),
                "workers_used": [r.specialty.value for r in valid_results],
                **combined.get("metadata", {})
            }
        )
    
    def reason(
        self,
        query: str,
        strategy: CombinationStrategy = CombinationStrategy.CONFIDENCE,
        context: Optional[Dict] = None,
        worker_filter: Optional[List[WorkerSpecialty]] = None
    ) -> MetaResult:
        """Synchronous version of reason_async.
        
        Args:
            query: The query to solve
            strategy: How to combine worker results
            context: Optional context for workers
            worker_filter: Only use workers with these specialties
            
        Returns:
            MetaResult with combined answer
        """
        return asyncio.run(
            self.reason_async(query, strategy, context, worker_filter)
        )
    
    def _combine_results(
        self,
        results: List[WorkerResult],
        strategy: CombinationStrategy,
        query: str
    ) -> Dict[str, Any]:
        """Combine worker results using specified strategy.
        
        Args:
            results: List of worker results
            strategy: Combination strategy
            query: Original query
            
        Returns:
            Dict with answer, confidence, reasoning, metadata
        """
        if strategy == CombinationStrategy.VOTING:
            return self._voting_strategy(results)
        elif strategy == CombinationStrategy.CONFIDENCE:
            return self._confidence_strategy(results)
        elif strategy == CombinationStrategy.VERIFICATION:
            return self._verification_strategy(results)
        elif strategy == CombinationStrategy.SYNTHESIS:
            return self._synthesis_strategy(results, query)
        elif strategy == CombinationStrategy.WEIGHTED:
            return self._weighted_strategy(results)
        else:
            # Default to confidence
            return self._confidence_strategy(results)
    
    def _voting_strategy(self, results: List[WorkerResult]) -> Dict[str, Any]:
        """Pick answer by majority vote.
        
        Args:
            results: Worker results
            
        Returns:
            Combined result dict
        """
        # Count votes for each answer
        answers = [r.answer for r in results]
        vote_counts = Counter(answers)
        
        # Get most common answer
        winner, votes = vote_counts.most_common(1)[0]
        
        # Find a result with the winning answer
        winning_result = next(r for r in results if r.answer == winner)
        
        # Calculate confidence based on vote proportion
        vote_proportion = votes / len(results)
        confidence = winning_result.confidence * vote_proportion
        
        return {
            "answer": winner,
            "confidence": confidence,
            "reasoning": f"Selected by majority vote ({votes}/{len(results)} workers agreed)",
            "metadata": {
                "votes": votes,
                "total_workers": len(results),
                "vote_proportion": vote_proportion
            }
        }
    
    def _confidence_strategy(self, results: List[WorkerResult]) -> Dict[str, Any]:
        """Pick result with highest confidence.
        
        Args:
            results: Worker results
            
        Returns:
            Combined result dict
        """
        # Sort by confidence
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)
        best = sorted_results[0]
        
        return {
            "answer": best.answer,
            "confidence": best.confidence,
            "reasoning": f"Selected highest confidence result from {best.specialty.value} worker (confidence: {best.confidence:.2f})",
            "metadata": {
                "best_worker": best.specialty.value,
                "all_confidences": [r.confidence for r in results]
            }
        }
    
    def _verification_strategy(self, results: List[WorkerResult]) -> Dict[str, Any]:
        """Pick result with best verification status.
        
        Args:
            results: Worker results
            
        Returns:
            Combined result dict
        """
        # Filter to verified results
        verified = [r for r in results if r.verification_passed]
        
        if not verified:
            # Fall back to confidence if nothing verified
            return self._confidence_strategy(results)
        
        # Among verified, pick highest confidence
        best = max(verified, key=lambda r: r.confidence)
        
        return {
            "answer": best.answer,
            "confidence": best.confidence * 1.1,  # Bonus for verification
            "reasoning": f"Selected verified result from {best.specialty.value} worker ({len(verified)}/{len(results)} workers verified)",
            "metadata": {
                "verified_count": len(verified),
                "total_workers": len(results),
                "best_worker": best.specialty.value
            }
        }
    
    def _weighted_strategy(self, results: List[WorkerResult]) -> Dict[str, Any]:
        """Weight results by specialty match and confidence.
        
        Args:
            results: Worker results
            
        Returns:
            Combined result dict
        """
        # Calculate weighted scores
        weighted_results = []
        for result in results:
            # Weight = confidence * verification_bonus
            weight = result.confidence
            if result.verification_passed:
                weight *= 1.2
            
            weighted_results.append((result, weight))
        
        # Sort by weight
        weighted_results.sort(key=lambda x: x[1], reverse=True)
        best_result, best_weight = weighted_results[0]
        
        return {
            "answer": best_result.answer,
            "confidence": best_result.confidence,
            "reasoning": f"Selected by weighted scoring (weight: {best_weight:.2f}) from {best_result.specialty.value} worker",
            "metadata": {
                "best_worker": best_result.specialty.value,
                "best_weight": best_weight,
                "all_weights": [w for _, w in weighted_results]
            }
        }
    
    def _synthesis_strategy(
        self,
        results: List[WorkerResult],
        query: str
    ) -> Dict[str, Any]:
        """Use LLM to synthesize multiple results into one.
        
        Args:
            results: Worker results
            query: Original query
            
        Returns:
            Combined result dict
        """
        if not self.llm_client:
            # Fall back to confidence if no LLM
            return self._confidence_strategy(results)
        
        # Build prompt with all worker results
        results_text = ""
        for i, result in enumerate(results, 1):
            results_text += f"\n{i}. {result.specialty.value.upper()} Worker (confidence {result.confidence:.2f}):\n"
            results_text += f"   Answer: {result.answer}\n"
            if result.reasoning_steps:
                results_text += f"   Reasoning: {'; '.join(result.reasoning_steps[:3])}\n"
        
        messages = [
            Message(
                role="system",
                content="You are a meta-reasoning expert. Synthesize multiple answers into the best single answer."
            ),
            Message(
                role="user",
                content=f"Query: {query}\n\nWorker Results:{results_text}\n\nSynthesize these results into the best answer. Be concise."
            )
        ]
        
        try:
            synthesized = self.llm_client.generate(messages)
            
            # Average confidence of all workers
            avg_confidence = sum(r.confidence for r in results) / len(results)
            
            return {
                "answer": synthesized,
                "confidence": min(avg_confidence * 1.1, 1.0),  # Bonus for synthesis
                "reasoning": f"Synthesized from {len(results)} workers using LLM",
                "metadata": {
                    "num_workers": len(results),
                    "workers": [r.specialty.value for r in results]
                }
            }
        except Exception as e:
            # LLM synthesis failed - use confidence-based selection instead
            result = self._confidence_strategy(results)
            result["metadata"]["synthesis_error"] = str(e)
            return result
