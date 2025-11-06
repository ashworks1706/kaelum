"""Core orchestration logic for Kaelum reasoning system.

Kaelum Architecture:
1. Router: Intelligently routes query to expert worker (math/logic/code/factual/creative/analysis)
2. Worker: Uses LATS (tree search) + caching for multi-step reasoning
3. Verification: Checks correctness (symbolic math, logic, etc.)
4. Reflection: If verification fails, improves reasoning and retries
5. Loop until verification passes or max iterations reached

Complete Flow:
Query → Router → Expert Worker (LATS + Cache) → Verification → Reflection (if failed) → Retry → Result
"""

import time
import logging
from typing import Iterator, Dict, Any, Optional, List

from core.config import KaelumConfig
from core.reasoning import LLMClient
from core.learning import CostTracker
from core.search import Router
from core.workers import WorkerSpecialty, create_worker
from core.search import TreeCache
from core.verification import VerificationEngine
from core.verification import ReflectionEngine
from core.learning import AdaptivePenalty
from core.learning import ActiveLearningEngine

logger = logging.getLogger("kaelum.orchestrator")


class KaelumOrchestrator:
    """Orchestrates complete reasoning pipeline with verification and reflection.
    
    Flow:
    1. Router selects expert worker
    2. Worker reasons using LATS + caching
    3. Verification checks correctness
    4. Reflection improves if verification fails
    5. Repeat until pass or max iterations
    """

    def __init__(self, config: KaelumConfig, reasoning_system_prompt=None, 
                 reasoning_user_template=None, enable_routing: bool = True,
                 enable_active_learning: bool = True,
                 cache_dir: str = ".kaelum/cache",
                 router_data_dir: str = ".kaelum/routing",
                 parallel: bool = False,
                 max_workers: int = 4,
                 max_tree_depth: Optional[int] = None,
                 num_simulations: Optional[int] = None,
                 router_learning_rate: float = 0.001,
                 router_buffer_size: int = 32,
                 router_exploration_rate: float = 0.1):
        self.config = config
        self.llm = LLMClient(config.reasoning_llm)
        self.metrics = CostTracker()
        self.tree_cache = TreeCache(
            cache_dir=cache_dir,
            embedding_model=config.embedding_model,
            llm_client=self.llm
        )
        
        self.router = Router(
            learning_enabled=True, 
            data_dir=router_data_dir,
            embedding_model=config.embedding_model,
            buffer_size=router_buffer_size,
            learning_rate=router_learning_rate,
            exploration_rate=router_exploration_rate
        ) if enable_routing else None
        if self.router:
            logger.info(f"Router enabled: Embedding-based intelligent routing ({config.embedding_model})")
            logger.info(f"  - Online learning: buffer_size={router_buffer_size}, lr={router_learning_rate}, exploration={router_exploration_rate}")
        
        self.verification_engine = VerificationEngine(
            self.llm,
            use_symbolic=config.use_symbolic_verification,
            use_factual=config.use_factual_verification,
            debug=config.debug_verification,
            embedding_model=config.embedding_model
        )
        self.reflection_engine = ReflectionEngine(
            self.llm,
            verification_engine=self.verification_engine,
            max_iterations=config.max_reflection_iterations
        )
        
        self.active_learning = ActiveLearningEngine(embedding_model=config.embedding_model) if enable_active_learning else None
        if self.active_learning:
            logger.info("Active learning enabled: Intelligent query selection for fine-tuning")
        
        self._workers = {}
        
        # Store runtime parameters from command line
        self.parallel = parallel
        self.max_workers = max_workers
        self.override_max_tree_depth = max_tree_depth  # User override from CLI
        self.override_num_simulations = num_simulations  # User override from CLI
        
        logger.info("=" * 70)
        logger.info("Kaelum Orchestrator Initialized")
        logger.info(f"  Embedding Model: {config.embedding_model}")
        logger.info(f"  Router: {'Enabled' if enable_routing else 'Disabled'}")
        if enable_routing:
            logger.info(f"    - Router data dir: {router_data_dir}")
        logger.info(f"  Tree Cache: Enabled")
        logger.info(f"    - Cache dir: {cache_dir}")
        logger.info(f"  Verification: Symbolic={config.use_symbolic_verification}, Factual={config.use_factual_verification}")
        logger.info(f"  Reflection: Max {config.max_reflection_iterations} iterations")
        logger.info(f"  Parallel LATS: {'Enabled' if parallel else 'Disabled'} (max {max_workers} workers)")
        logger.info("=" * 70)

    def _get_worker(self, specialty: str):
        if specialty not in self._workers:
            try:
                specialty_enum = WorkerSpecialty(specialty)
                self._workers[specialty] = create_worker(specialty_enum, self.config, tree_cache=self.tree_cache)
            except Exception as e:
                logger.error(f"Failed to create {specialty} worker: {e}")
                self._workers[specialty] = create_worker(WorkerSpecialty.LOGIC, self.config, tree_cache=self.tree_cache)
        
        return self._workers[specialty]
    
    def infer(self, query: str, stream: bool = False):
        """Run complete reasoning pipeline with verification and reflection.
        
        Architecture:
        1. Router → Select expert worker based on query type
        2. Worker → Use LATS tree search + caching for reasoning
        3. Verification → Check if reasoning is correct
        4. Reflection → If verification fails, improve and retry
        5. Loop until verification passes or max iterations
        
        Args:
            query: User's question
            stream: Streaming not yet supported
            
        Returns:
            Dictionary with answer, reasoning steps, verification status, etc.
        """
        if stream:
            logger.warning("Streaming not yet supported, using sync mode")
            stream = False
        
        logger.info("=" * 70)
        logger.info(f"QUERY: {query}")
        logger.info("=" * 70)
        
        # Step 1: Check cache FIRST (before routing/detectors)
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(self.config.embedding_model)
        query_embedding = encoder.encode(query)
        
        cached_tree = self.tree_cache.get(query, query_embedding, similarity_threshold=0.85)
        if cached_tree and cached_tree.get("quality") == "high":
            logger.info("CACHE: ✓ HIT (high quality) - returning cached result")
            cache_result = cached_tree["result"]
            cache_result["cache_hit"] = True
            cache_result["metrics"]["total_time_ms"] = 1
            return cache_result
        
        # Step 2: Route query to appropriate expert worker (only on cache miss)
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: ROUTING TO EXPERT WORKER")
        logger.info("=" * 70)
        if self.router:
            routing_decision = self.router.route(query)
            worker_specialty = routing_decision.worker_specialty
            use_cache = False
            max_depth = routing_decision.max_tree_depth
            num_sims = routing_decision.num_simulations
            logger.info(f"ROUTING: ✓ Selected {worker_specialty.upper()} worker")
            logger.info(f"ROUTING: Confidence = {routing_decision.confidence:.3f}")
            logger.info(f"ROUTING: Complexity = {routing_decision.complexity_score:.3f}")
        else:
            worker_specialty = "logic"
            use_cache = False
            max_depth = 5
            num_sims = 10
            logger.info(f"ROUTING: Default to {worker_specialty.upper()} worker (router disabled)")
        
        # Apply CLI overrides if provided
        if self.override_max_tree_depth is not None:
            logger.info(f"ROUTING: Overriding max_tree_depth: {max_depth} → {self.override_max_tree_depth} (from CLI)")
            max_depth = self.override_max_tree_depth
        
        if self.override_num_simulations is not None:
            logger.info(f"ROUTING: Overriding num_simulations: {num_sims} → {self.override_num_simulations} (from CLI)")
            num_sims = self.override_num_simulations
        
        worker = self._get_worker(worker_specialty)
        
        # Router already provides optimal params via neural network
        # No need for separate adaptive config
        
        session_id = f"session_{int(time.time() * 1000)}"
        self.metrics.start_session(session_id, metadata={"query": query[:50]})
        start_time = time.time()
        
        max_iterations = self.config.max_reflection_iterations + 1  # Initial attempt + reflections
        iteration = 0
        verification_passed = False
        final_result = None
        
        while iteration < max_iterations and not verification_passed:
            iteration += 1
            logger.info(f"\n{'=' * 70}")
            logger.info(f"ITERATION {iteration}/{max_iterations}")
            logger.info(f"{'=' * 70}")
            
            # Step 2: Worker reasons using LATS + caching
            logger.info("\n" + "=" * 70)
            logger.info(f"STEP 3: WORKER EXECUTION ({worker_specialty.upper()})")
            logger.info("=" * 70)
            logger.info(f"WORKER: Executing {worker_specialty.upper()} worker with LATS tree search")
            logger.info(f"  - Max tree depth: {max_depth}")
            logger.info(f"  - Num simulations: {num_sims}")
            logger.info(f"  - Parallel: {self.parallel}")
            if self.parallel:
                logger.info(f"  - Max workers: {self.max_workers}")
            logger.info(f"  - Use cache: {use_cache}")
            
            worker_start_time = time.time()
            result = worker.solve(
                query,
                context=None,
                use_cache=False,
                max_tree_depth=max_depth,
                num_simulations=num_sims,
                parallel=self.parallel,
                max_workers=self.max_workers
            )
            worker_time = time.time() - worker_start_time
            
            logger.info(f"WORKER: ✓ Execution complete in {worker_time:.2f}s")
            logger.info(f"WORKER: Generated answer with {len(result.reasoning_steps)} reasoning steps")
            logger.info(f"WORKER: Confidence = {result.confidence:.3f}")
            logger.info(f"WORKER: Answer preview: {result.answer[:100]}...")
            
            # Step 3: Verification - check if reasoning is correct
            logger.info("\n" + "=" * 70)
            logger.info("STEP 4: VERIFICATION")
            logger.info("=" * 70)
            logger.info(f"VERIFICATION: Checking reasoning correctness for {worker_specialty.upper()} worker output")
            
            verification_start_time = time.time()
            verification_result = self.verification_engine.verify(
                query=query,
                reasoning_steps=result.reasoning_steps,
                answer=result.answer,
                worker_type=worker_specialty
            )
            verification_time = time.time() - verification_start_time
            
            verification_passed = verification_result["passed"]
            confidence = verification_result["confidence"]
            issues = verification_result.get("issues", [])
            
            logger.info(f"VERIFICATION: Completed in {verification_time:.2f}s")
            
            if verification_passed:
                logger.info(f"VERIFICATION: ✓ PASSED")
                logger.info(f"  - Confidence: {confidence:.3f}")
                logger.info(f"  - Symbolic check: {verification_result.get('symbolic_passed', 'N/A')}")
                logger.info(f"  - Factual check: {verification_result.get('factual_passed', 'N/A')}")
                final_result = result
                final_result.verification_passed = True
                final_result.confidence = confidence
            else:
                logger.info(f"VERIFICATION: ✗ FAILED")
                logger.info(f"  - Confidence: {confidence:.3f}")
                if issues:
                    logger.info(f"  - Issues found ({len(issues)}):")
                    for i, issue in enumerate(issues, 1):
                        logger.info(f"    {i}. {issue}")
                
                # Step 4: Reflection - improve reasoning if not last iteration
                if iteration < max_iterations:
                    logger.info("\n" + "=" * 70)
                    logger.info("STEP 5: REFLECTION")
                    logger.info("=" * 70)
                    logger.info(f"REFLECTION: Improving reasoning based on verification failures")
                    logger.info(f"REFLECTION: Issues to address: {len(issues)}")
                    
                    reflection_start_time = time.time()
                    improved_steps = self.reflection_engine.enhance_reasoning(
                        query=query,
                        initial_trace=result.reasoning_steps,
                        worker_type=worker_specialty,
                        verification_issues=issues
                    )
                    reflection_time = time.time() - reflection_start_time
                    
                    # Update worker's reasoning for next iteration
                    # (Next iteration will generate new answer based on improved understanding)
                    logger.info(f"REFLECTION: ✓ Completed in {reflection_time:.2f}s")
                    logger.info(f"REFLECTION: Generated {len(improved_steps)} improved reasoning steps")
                    logger.info(f"REFLECTION: Will retry with iteration {iteration + 1}/{max_iterations}")
                else:
                    logger.info("\n" + "=" * 70)
                    logger.info("REFLECTION: Max iterations reached, using best attempt")
                    logger.info("=" * 70)
                    final_result = result
                    final_result.verification_passed = False
                    final_result.confidence = confidence
        
        total_time = time.time() - start_time
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"COMPLETED: {iteration} iteration(s), verification {'PASSED' if verification_passed else 'FAILED'}")
        logger.info(f"TIME: {total_time:.3f}s")
        logger.info(f"{'=' * 70}\n")
        
        # Format response
        response = {
            "query": query,
            "reasoning_trace": final_result.reasoning_steps,
            "answer": final_result.answer,
            "worker": worker_specialty,
            "confidence": final_result.confidence,
            "verification_passed": verification_passed,
            "iterations": iteration,
            "cache_hit": False,
            "metrics": {
                "total_time_ms": total_time * 1000,
                "execution_time_ms": final_result.execution_time * 1000,
                "tree_depth": final_result.metadata.get("tree_depth", 0),
                "num_simulations": final_result.metadata.get("num_simulations", 0),
                "iterations": iteration
            }
        }
        
        # Store in cache with quality metadata
        quality = "high" if verification_passed and final_result.confidence > 0.7 else "low"
        self.tree_cache.store(
            query_embedding,
            {
                "result": response,
                "quality": quality,
                "confidence": final_result.confidence,
                "worker": worker_specialty
            }
        )
        logger.info(f"CACHE: Stored result with quality={quality}")
        
        # Step 6: Record outcome for router learning with enhanced feedback
        if self.router:
            avg_reward = final_result.metadata.get("avg_reward", final_result.confidence)
            actual_depth = final_result.metadata.get("tree_depth", max_depth)
            actual_sims = final_result.metadata.get("num_simulations", num_sims)
            
            outcome = {
                "query": query,
                "success": verification_passed,
                "confidence": final_result.confidence,
                "execution_time": total_time,
                "cost": total_time * 0.00000001,
                "verification_passed": verification_passed,
                "avg_reward": avg_reward,
                "predicted_depth": max_depth,
                "actual_depth": actual_depth,
                "predicted_sims": num_sims,
                "actual_sims": actual_sims,
                "cache_quality": quality
            }
            self.router.record_outcome(routing_decision, outcome)
        
        # Format response
        response = {
            "query": query,
            "reasoning_trace": final_result.reasoning_steps,
            "answer": final_result.answer,
            "worker": worker_specialty,
            "confidence": final_result.confidence,
            "verification_passed": verification_passed,
            "iterations": iteration,
            "cache_hit": final_result.metadata.get("cache_hit", False),
            "metrics": {
                "total_time_ms": total_time * 1000,
                "execution_time_ms": final_result.execution_time * 1000,
                "tree_depth": final_result.metadata.get("tree_depth", 0),
                "num_simulations": final_result.metadata.get("num_simulations", 0),
                "iterations": iteration
            }
        }
        
        # Log metrics with token counting
        input_text = query
        output_text = final_result.answer + " ".join(final_result.reasoning_steps[:3])
        self.metrics.log_inference(
            input_text=input_text,
            output_text=output_text,
            latency_ms=total_time * 1000,
            worker_type=worker_specialty,
            verification_passed=verification_passed,
            cache_hit=final_result.metadata.get("cache_hit", False),
            num_simulations=final_result.metadata.get("num_simulations", 0),
            session_id=session_id
        )
        
        # Collect for active learning
        if self.active_learning:
            self.active_learning.collect_query(query, response)
        
        return response
    
    def get_metrics_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        session_metrics = self.metrics.get_session_metrics(session_id)
        analytics = self.metrics.get_analytics_summary()
        
        return {
            "session": session_metrics,
            "analytics": analytics
        }
    
    def get_active_learning_stats(self) -> Dict[str, Any]:
        """Get active learning statistics."""
        if not self.active_learning:
            return {"active_learning": "disabled"}
        return self.active_learning.get_statistics()
    
    def generate_training_batch(
        self,
        strategy: str = "mixed",
        batch_size: int = 20
    ) -> List[Dict[str, Any]]:
        """Generate training batch using active learning.
        
        Args:
            strategy: Selection strategy (uncertainty, diversity, error, complexity, mixed)
            batch_size: Number of queries to select
        
        Returns:
            List of training examples
        """
        if not self.active_learning:
            logger.warning("Active learning is disabled")
            return []
        
        return self.active_learning.generate_training_batch(strategy, batch_size)
    
    def export_training_dataset(self, output_path: str) -> int:
        """Export collected training data.
        
        Args:
            output_path: Path to save training dataset
        
        Returns:
            Number of examples exported
        """
        if not self.active_learning:
            logger.warning("Active learning is disabled")
            return 0
        
        return self.active_learning.export_training_dataset(output_path)
