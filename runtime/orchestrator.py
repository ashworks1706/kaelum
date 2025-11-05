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
                 enable_active_learning: bool = True):
        self.config = config
        self.llm = LLMClient(config.reasoning_llm)
        self.metrics = CostTracker()
        self.tree_cache = TreeCache()
        
        self.router = Router(learning_enabled=True) if enable_routing else None
        if self.router:
            logger.info("Router enabled: Embedding-based intelligent routing")
        
        self.verification_engine = VerificationEngine(
            self.llm,
            use_symbolic=config.use_symbolic_verification,
            use_factual=config.use_factual_verification,
            debug=config.debug_verification
        )
        self.reflection_engine = ReflectionEngine(
            self.llm,
            verification_engine=self.verification_engine,
            max_iterations=config.max_reflection_iterations
        )
        
        self.active_learning = ActiveLearningEngine() if enable_active_learning else None
        if self.active_learning:
            logger.info("Active learning enabled: Intelligent query selection for fine-tuning")
        
        self._workers = {}
        
        logger.info("=" * 70)
        logger.info("Kaelum Orchestrator Initialized")
        logger.info(f"  Router: {'Enabled' if enable_routing else 'Disabled'}")
        logger.info(f"  Verification: Symbolic={config.use_symbolic_verification}, Factual={config.use_factual_verification}")
        logger.info(f"  Reflection: Max {config.max_reflection_iterations} iterations")
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
        
        # Step 1: Route query to appropriate expert worker
        if self.router:
            routing_decision = self.router.route(query) # neural network to extract keywords and semantic depth to determine complexity of the query
            worker_specialty = routing_decision.worker_specialty
            use_cache = routing_decision.use_tree_cache
            max_depth = routing_decision.max_tree_depth
            num_sims = routing_decision.num_simulations
            logger.info(f"ROUTING: Selected {worker_specialty} worker")
        else:
            # Default routing
            worker_specialty = "logic"
            use_cache = True
            max_depth = 5
            num_sims = 10
            logger.info(f"ROUTING: Default to {worker_specialty} worker (router disabled)")
        
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
            logger.info(f"WORKER: {worker_specialty} executing with LATS")
            logger.info(f"  Config: depth={max_depth}, sims={num_sims}, cache={use_cache}")
            
            result = worker.solve(
                query,
                context=None,
                use_cache=use_cache and (iteration == 1),  # Only use cache on first attempt
                max_tree_depth=max_depth,
                num_simulations=num_sims
            )
            
            logger.info(f"WORKER: Generated answer with {len(result.reasoning_steps)} reasoning steps")
            logger.info(f"WORKER: Confidence = {result.confidence:.2f}")
            
            # Step 3: Verification - check if reasoning is correct
            logger.info(f"\nVERIFICATION: Checking reasoning correctness...")
            verification_result = self.verification_engine.verify(
                query=query,
                reasoning_steps=result.reasoning_steps,
                answer=result.answer,
                worker_type=worker_specialty
            )
            
            verification_passed = verification_result["passed"]
            confidence = verification_result["confidence"]
            issues = verification_result.get("issues", [])
            
            if verification_passed:
                logger.info(f"VERIFICATION: ✓ PASSED (confidence={confidence:.2f})")
                final_result = result
                final_result.verification_passed = True
                final_result.confidence = confidence
            else:
                logger.info(f"VERIFICATION: ✗ FAILED (confidence={confidence:.2f})")
                if issues:
                    logger.info(f"VERIFICATION: Issues found:")
                    for issue in issues:
                        logger.info(f"  - {issue}")
                
                # Step 4: Reflection - improve reasoning if not last iteration
                if iteration < max_iterations:
                    logger.info(f"\nREFLECTION: Improving reasoning based on verification failures...")
                    improved_steps = self.reflection_engine.enhance_reasoning(
                        query=query,
                        initial_trace=result.reasoning_steps,
                        worker_type=worker_specialty,
                        verification_issues=issues
                    )
                    
                    # Update worker's reasoning for next iteration
                    # (Next iteration will generate new answer based on improved understanding)
                    logger.info(f"REFLECTION: Generated {len(improved_steps)} improved reasoning steps")
                else:
                    logger.info(f"REFLECTION: Max iterations reached, using best attempt")
                    final_result = result
                    final_result.verification_passed = False
                    final_result.confidence = confidence
        
        total_time = time.time() - start_time
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"COMPLETED: {iteration} iteration(s), verification {'PASSED' if verification_passed else 'FAILED'}")
        logger.info(f"TIME: {total_time:.3f}s")
        logger.info(f"{'=' * 70}\n")
        
        # Step 6: Record outcome for router learning
        if self.router:
            outcome = {
                "query": query,
                "success": verification_passed,
                "confidence": final_result.confidence,
                "execution_time": total_time,
                "cost": total_time * 0.00000001,
                "verification_passed": verification_passed
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
