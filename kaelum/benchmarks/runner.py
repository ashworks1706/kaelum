"""Benchmark runner to execute queries and collect results."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from kaelum.benchmarks.dataset import BenchmarkQuery, BenchmarkDataset
from kaelum.core.workers import WorkerAgent, create_worker, WorkerSpecialty
from kaelum.core.meta_reasoner import MetaReasoner, CombinationStrategy


class RunMode(Enum):
    """Benchmark run modes."""
    SINGLE_WORKER = "single_worker"  # Use one worker per query
    META_REASONER = "meta_reasoner"  # Use meta-reasoner with multiple workers
    COMPARISON = "comparison"  # Run both and compare


@dataclass
class BenchmarkResult:
    """Result from running a single benchmark query.
    
    Attributes:
        query_id: Query identifier
        query: Original query text
        answer: Generated answer
        execution_time: Time in seconds
        confidence: Confidence score (0-1)
        verified: Whether answer was verified
        mode: Run mode used
        workers_used: Which workers were involved
        metadata: Additional information
    """
    query_id: str
    query: str
    answer: str
    execution_time: float
    confidence: float
    verified: bool
    mode: RunMode
    workers_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "answer": self.answer,
            "execution_time": self.execution_time,
            "confidence": self.confidence,
            "verified": self.verified,
            "mode": self.mode.value,
            "workers_used": self.workers_used,
            "metadata": self.metadata
        }


class BenchmarkRunner:
    """Executes benchmark queries and collects results."""
    
    def __init__(self, llm_client):
        """Initialize runner.
        
        Args:
            llm_client: LLM client for workers
        """
        self.llm_client = llm_client
        self.results: List[BenchmarkResult] = []
    
    async def run_single_worker(
        self,
        query: BenchmarkQuery,
        worker_specialty: WorkerSpecialty
    ) -> BenchmarkResult:
        """Run query with a single worker.
        
        Args:
            query: Query to execute
            worker_specialty: Which worker to use
            
        Returns:
            Benchmark result
        """
        worker = create_worker(worker_specialty, self.llm_client)
        
        start_time = time.time()
        result = await worker.process(query.query)
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            query_id=query.id,
            query=query.query,
            answer=result.answer,
            execution_time=execution_time,
            confidence=result.confidence,
            verified=result.verification_passed,
            mode=RunMode.SINGLE_WORKER,
            workers_used=[worker_specialty.value],
            metadata={
                "category": query.category.value,
                "difficulty": query.difficulty.value,
                "reasoning_steps": result.reasoning_steps[:3] if result.reasoning_steps else None  # First 3 steps
            }
        )
    
    async def run_meta_reasoner(
        self,
        query: BenchmarkQuery,
        strategy: CombinationStrategy,
        workers: Optional[List[WorkerAgent]] = None
    ) -> BenchmarkResult:
        """Run query with meta-reasoner.
        
        Args:
            query: Query to execute
            strategy: Combination strategy
            workers: Optional list of workers (auto-selected if None)
            
        Returns:
            Benchmark result
        """
        # Create meta-reasoner
        meta = MetaReasoner(self.llm_client)
        
        # Use provided workers or create default set
        if workers is None:
            workers = [
                create_worker(WorkerSpecialty.MATH, self.llm_client),
                create_worker(WorkerSpecialty.LOGIC, self.llm_client),
            ]
        
        # Add workers to meta-reasoner
        meta.add_workers(workers)
        
        start_time = time.time()
        result = await meta.reason_async(query.query, strategy=strategy)
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            query_id=query.id,
            query=query.query,
            answer=result.answer,
            execution_time=execution_time,
            confidence=result.confidence,
            verified=all(r.verification_passed for r in result.worker_results),
            mode=RunMode.META_REASONER,
            workers_used=[r.specialty.value for r in result.worker_results],
            metadata={
                "category": query.category.value,
                "difficulty": query.difficulty.value,
                "strategy": strategy.value,
                "num_workers": len(result.worker_results),
                "reasoning": result.reasoning[:200] if result.reasoning else None  # Truncate
            }
        )
    
    async def run_comparison(
        self,
        query: BenchmarkQuery,
        worker_specialty: WorkerSpecialty,
        strategy: CombinationStrategy
    ) -> Dict[str, BenchmarkResult]:
        """Run query with both single worker and meta-reasoner.
        
        Args:
            query: Query to execute
            worker_specialty: Worker for single mode
            strategy: Strategy for meta mode
            
        Returns:
            Dictionary with both results
        """
        # Run both in parallel
        single_task = self.run_single_worker(query, worker_specialty)
        meta_task = self.run_meta_reasoner(query, strategy)
        
        single_result, meta_result = await asyncio.gather(single_task, meta_task)
        
        return {
            "single": single_result,
            "meta": meta_result
        }
    
    async def run_dataset(
        self,
        dataset: BenchmarkDataset,
        mode: RunMode = RunMode.META_REASONER,
        strategy: CombinationStrategy = CombinationStrategy.WEIGHTED,
        max_queries: Optional[int] = None
    ) -> List[BenchmarkResult]:
        """Run all queries in dataset.
        
        Args:
            dataset: Dataset to run
            mode: Run mode
            strategy: Combination strategy (for meta mode)
            max_queries: Maximum queries to run (None = all)
            
        Returns:
            List of results
        """
        queries = list(dataset.queries)
        if max_queries:
            queries = queries[:max_queries]
        
        results = []
        
        for query in queries:
            try:
                if mode == RunMode.SINGLE_WORKER:
                    # Choose worker based on category
                    specialty = self._get_specialty_for_category(query.category.value)
                    result = await self.run_single_worker(query, specialty)
                    
                elif mode == RunMode.META_REASONER:
                    result = await self.run_meta_reasoner(query, strategy)
                    
                elif mode == RunMode.COMPARISON:
                    specialty = self._get_specialty_for_category(query.category.value)
                    comparison = await self.run_comparison(query, specialty, strategy)
                    # Store both results
                    results.append(comparison["single"])
                    results.append(comparison["meta"])
                    continue
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing query {query.id}: {e}")
                # Add error result
                results.append(BenchmarkResult(
                    query_id=query.id,
                    query=query.query,
                    answer=f"ERROR: {str(e)}",
                    execution_time=0.0,
                    confidence=0.0,
                    verified=False,
                    mode=mode,
                    metadata={"error": str(e)}
                ))
        
        self.results.extend(results)
        return results
    
    def _get_specialty_for_category(self, category: str) -> WorkerSpecialty:
        """Map query category to worker specialty."""
        mapping = {
            "math": WorkerSpecialty.MATH,
            "logic": WorkerSpecialty.LOGIC,
            "code": WorkerSpecialty.CODE,
            "factual": WorkerSpecialty.FACTUAL,
            "creative": WorkerSpecialty.CREATIVE,
            "mixed": WorkerSpecialty.ANALYSIS  # Default for mixed
        }
        return mapping.get(category, WorkerSpecialty.LOGIC)
    
    def get_results_by_mode(self, mode: RunMode) -> List[BenchmarkResult]:
        """Filter results by run mode."""
        return [r for r in self.results if r.mode == mode]
    
    def get_results_by_query_id(self, query_id: str) -> List[BenchmarkResult]:
        """Get all results for a specific query."""
        return [r for r in self.results if r.query_id == query_id]
    
    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all results."""
        if not self.results:
            return {}
        
        return {
            "total_queries": len(self.results),
            "avg_execution_time": sum(r.execution_time for r in self.results) / len(self.results),
            "avg_confidence": sum(r.confidence for r in self.results) / len(self.results),
            "verified_count": sum(1 for r in self.results if r.verified),
            "by_mode": {
                mode.value: len([r for r in self.results if r.mode == mode])
                for mode in RunMode
            }
        }
