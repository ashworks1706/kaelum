"""Core orchestration logic for Kaelum reasoning system.

Architecture:
1. Router → Select expert worker based on query type
2. Worker → Use LATS tree search + caching for reasoning
3. Verification → Check if reasoning is correct
4. Reflection → If verification fails, improve and retry
5. Loop until verification passes or max iterations

Flow: Query → Router → Worker (LATS + Cache) → Verification → Reflection (if failed) → Retry → Result
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
from core.verification import ReflectionEngine
from core.learning import ActiveLearningEngine
from core.shared_encoder import get_shared_encoder
from core.paths import DEFAULT_CACHE_DIR, DEFAULT_ROUTER_DIR

logger = logging.getLogger("kaelum.orchestrator")

class KaelumOrchestrator:
    """Orchestrates complete reasoning pipeline with verification and reflection."""

    def __init__(self, config: KaelumConfig, reasoning_system_prompt=None, 
                 reasoning_user_template=None, enable_routing: bool = True,
                 enable_active_learning: bool = True,
                 cache_dir: str = DEFAULT_CACHE_DIR,
                 router_data_dir: str = DEFAULT_ROUTER_DIR,
                 parallel: bool = False,
                 max_workers: int = 4,
                 max_tree_depth: Optional[int] = None,
                 num_simulations: Optional[int] = None,
                 router_learning_rate: float = 0.001,
                 router_buffer_size: int = 32,
                 router_exploration_rate: float = 0.1,
                 router_depth_min: int = 3,
                 router_depth_max: int = 10,
                 router_sims_min: int = 5,
                 router_sims_max: int = 25,
                 lats_exploration_constant: float = 1.414,
                 lats_prune_visit_threshold: int = 3,
                 lats_prune_reward_threshold: float = 0.3):
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
            exploration_rate=router_exploration_rate,
            depth_range=(router_depth_min, router_depth_max),
            sims_range=(router_sims_min, router_sims_max)
        ) if enable_routing else None
        if self.router:
            logger.info(f"Router enabled: Embedding-based intelligent routing ({config.embedding_model})")
            logger.info(f"  - Online learning: buffer_size={router_buffer_size}, lr={router_learning_rate}, exploration={router_exploration_rate}")
        
        self.reflection_engine = ReflectionEngine(
            self.llm,
            verification_engine=None,
            max_iterations=config.max_reflection_iterations
        )
        
        self.active_learning = ActiveLearningEngine(embedding_model=config.embedding_model) if enable_active_learning else None
        if self.active_learning:
            logger.info("Active learning enabled: Intelligent query selection for fine-tuning")
        
        self._workers = {}
        self._encoder = None

        # Expose a single shared feedback engine; reuse the router's if available
        from core.learning.human_feedback import HumanFeedbackEngine
        self.feedback_engine: HumanFeedbackEngine = (
            self.router.feedback_engine if self.router else HumanFeedbackEngine()
        )

        self.parallel = parallel
        self.max_workers = max_workers
        self.override_max_tree_depth = max_tree_depth
        self.override_num_simulations = num_simulations
        self.lats_params = {
            "exploration_constant": lats_exploration_constant,
            "prune_visit_threshold": lats_prune_visit_threshold,
            "prune_reward_threshold": lats_prune_reward_threshold
        }
        
        logger.info("=" * 70)
        logger.info("Kaelum Orchestrator Initialized")
        logger.info(f"  Embedding Model: {config.embedding_model}")
        logger.info(f"  Router: {'Enabled' if enable_routing else 'Disabled'}")
        if enable_routing:
            logger.info(f"    - Router data dir: {router_data_dir}")
        logger.info(f"  Tree Cache: Enabled")
        logger.info(f"    - Cache dir: {cache_dir}")
        logger.info(f"  PRM gate threshold: {config.prm_pass_threshold}")
        logger.info(f"  Reflection: Max {config.max_reflection_iterations} iterations")
        logger.info(f"  Parallel LATS: {'Enabled' if parallel else 'Disabled'} (max {max_workers} workers)")
        logger.info("=" * 70)

    def _infer_stream(self, query: str):
        """Run the full reasoning pipeline then stream the final answer token-by-token."""
        from core.reasoning import Message

        # Run the full pipeline synchronously first (routing, LATS, verification)
        result = self.infer(query, stream=False)

        # Re-stream the final answer via the LLM using the verified reasoning as context
        reasoning_context = "\n".join(
            f"{i+1}. {step}" for i, step in enumerate(result.get("reasoning_trace", []))
        )
        messages = [
            Message(
                role="system",
                content=(
                    "You are a precise, helpful assistant. "
                    "Using the verified reasoning steps provided, give the final answer concisely and clearly."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Query: {query}\n\n"
                    + (f"Reasoning:\n{reasoning_context}\n\n" if reasoning_context else "")
                    + "Final answer:"
                ),
            ),
        ]
        yield from self.llm.generate(messages, stream=True)

    def _get_worker(self, specialty: str):
        if specialty not in self._workers:
            specialty_enum = WorkerSpecialty(specialty)
            self._workers[specialty] = create_worker(specialty_enum, self.config, tree_cache=self.tree_cache)
        
        return self._workers[specialty]
    
    def infer(self, query: str, stream: bool = False):
        """Run complete reasoning pipeline with verification and reflection."""
        if stream:
            return self._infer_stream(query)
        
        logger.info("=" * 70)
        logger.info(f"QUERY: {query}")
        logger.info("=" * 70)
        
        if self._encoder is None:
            self._encoder = get_shared_encoder(self.config.embedding_model, device='cpu')
        
        query_embedding = self._encoder.encode(query)
        
        cached_tree = self.tree_cache.get(query, query_embedding, similarity_threshold=0.85)
        if cached_tree and cached_tree.get("quality") == "high":
            logger.info("CACHE: HIT (high quality) - returning cached result")
            cache_result = cached_tree["result"]
            cache_result["cache_hit"] = True
            cache_result["metrics"]["total_time_ms"] = 1
            return cache_result
        
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: ROUTING TO EXPERT WORKER")
        logger.info("=" * 70)
        if self.router:
            routing_decision = self.router.route(query)
            worker_specialty = routing_decision.worker_specialty
            use_cache = True
            max_depth = routing_decision.max_tree_depth
            num_sims = routing_decision.num_simulations
            router_confidence = routing_decision.confidence
            logger.info(f"ROUTING: Selected {worker_specialty.upper()} worker")
            logger.info(f"ROUTING: Confidence = {routing_decision.confidence:.3f}")
            logger.info(f"ROUTING: Complexity = {routing_decision.complexity_score:.3f}")
            
            use_ensemble = router_confidence < 0.6
            if use_ensemble:
                logger.info(f"ROUTING: ⚠ Low confidence ({router_confidence:.3f} < 0.6) - using ensemble voting")
        else:
            worker_specialty = "logic"
            use_cache = True
            max_depth = 5
            num_sims = 10
            router_confidence = 1.0
            use_ensemble = False
            logger.info(f"ROUTING: Default to {worker_specialty.upper()} worker (router disabled)")
        
        if self.override_max_tree_depth is not None:
            logger.info(f"ROUTING: Overriding max_tree_depth: {max_depth} → {self.override_max_tree_depth} (from CLI)")
            max_depth = self.override_max_tree_depth
        
        if self.override_num_simulations is not None:
            logger.info(f"ROUTING: Overriding num_simulations: {num_sims} → {self.override_num_simulations} (from CLI)")
            num_sims = self.override_num_simulations
        
        worker = self._get_worker(worker_specialty)
        
        if use_ensemble:
            ensemble_workers = []
            if worker_specialty == "math":
                ensemble_workers = ["math", "logic", "analysis"]
            elif worker_specialty == "code":
                ensemble_workers = ["code", "logic", "analysis"]
            elif worker_specialty == "creative":
                ensemble_workers = ["creative", "analysis"]
            else:
                ensemble_workers = [worker_specialty, "logic", "analysis"]
            
            logger.info(f"ENSEMBLE: Using {len(ensemble_workers)} workers for voting: {ensemble_workers}")
        else:
            ensemble_workers = None
        
        session_id = f"session_{int(time.time() * 1000)}"
        self.metrics.start_session(session_id, metadata={"query": query[:50]})
        start_time = time.time()
        
        max_iterations = self.config.max_reflection_iterations + 1
        iteration = 0
        verification_passed = False
        final_result = None
        # Tree reuse state — carry the LATS tree between reflection iterations so
        # MCTS can continue from where it left off instead of restarting cold.
        previous_tree = None
        prev_issues: list = []
        
        while iteration < max_iterations and not verification_passed:
            iteration += 1
            logger.info(f"\n{'=' * 70}")
            logger.info(f"ITERATION {iteration}/{max_iterations}")
            logger.info(f"{'=' * 70}")
            
            logger.info("\n" + "=" * 70)
            logger.info(f"STEP 3: WORKER EXECUTION ({worker_specialty.upper()})")
            logger.info("=" * 70)
            
            if ensemble_workers:
                logger.info(f"ENSEMBLE: Running {len(ensemble_workers)} workers in parallel")
                
                from concurrent.futures import ThreadPoolExecutor, as_completed
                results = []
                
                with ThreadPoolExecutor(max_workers=len(ensemble_workers)) as executor:
                    future_to_worker = {}
                    for worker_type in ensemble_workers:
                        logger.info(f"ENSEMBLE: Submitting {worker_type.upper()} worker")
                        worker_obj = self._get_worker(worker_type)
                        
                        # Use reduced simulations per worker in ensemble mode to avoid N×latency
                        ensemble_sims = max(3, num_sims // 2)
                        future = executor.submit(
                            worker_obj.solve,
                            query,
                            None,
                            use_cache,
                            max_depth,
                            ensemble_sims,
                            self.parallel,
                            self.max_workers
                        )
                        future_to_worker[future] = worker_type
                    
                    for future in as_completed(future_to_worker):
                        worker_type = future_to_worker[future]
                        try:
                            worker_result = future.result()
                            results.append((worker_type, worker_result))
                            logger.info(f"ENSEMBLE: {worker_type.upper()} completed (confidence={worker_result.confidence:.3f})")
                        except Exception as e:
                            logger.error(f"ENSEMBLE: {worker_type.upper()} failed: {e}")
                
                if not results:
                    raise RuntimeError("ENSEMBLE: All workers failed — no results to select from")
                
                results.sort(key=lambda x: x[1].confidence, reverse=True)
                
                logger.info(f"ENSEMBLE: Voting results:")
                for worker_type, res in results:
                    logger.info(f"  - {worker_type.upper()}: confidence={res.confidence:.3f}")
                
                best_worker, result = results[0]
                logger.info(f"ENSEMBLE: Selected {best_worker.upper()} result (highest confidence)")
                worker_specialty = best_worker
            else:
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
                    use_cache=use_cache,
                    max_tree_depth=max_depth,
                    num_simulations=num_sims,
                    parallel=self.parallel,
                    max_workers=self.max_workers,
                    existing_tree=previous_tree,
                    extra_sims=max(3, num_sims // 3),
                    verification_issues=prev_issues,
                )
                worker_time = time.time() - worker_start_time
                
                logger.info(f"WORKER: Execution complete in {worker_time:.2f}s")
            logger.info(f"WORKER: Generated answer with {len(result.reasoning_steps)} reasoning steps")
            logger.info(f"WORKER: Confidence = {result.confidence:.3f}")
            
            if result.answer:
                preview = result.answer[:100] if len(result.answer) > 100 else result.answer
                logger.info(f"WORKER: Answer preview: {preview}...")
            else:
                logger.warning(f"WORKER: No answer generated (answer is None or empty)")
            
            logger.info("\n" + "=" * 70)
            logger.info("STEP 4: PRM GATE")
            logger.info("=" * 70)
            logger.info(f"PRM GATE: Scoring {len(result.reasoning_steps)} steps for {worker_specialty.upper()} worker")

            from core.verification.process_reward_model import get_prm as _get_prm_gate
            _prm_gate = _get_prm_gate(self.config.embedding_model)
            prm_scores = [
                _prm_gate.predict_step_quality(
                    query, step, result.reasoning_steps[:i], worker_specialty
                )
                for i, step in enumerate(result.reasoning_steps)
            ] if result.reasoning_steps else []
            avg_prm = sum(prm_scores) / len(prm_scores) if prm_scores else 0.5
            verification_passed = avg_prm >= self.config.prm_pass_threshold
            confidence = avg_prm
            issues = [] if verification_passed else [
                f"PRM avg score {avg_prm:.3f} below threshold {self.config.prm_pass_threshold}"
            ]
            logger.info(f"PRM GATE: avg={avg_prm:.3f}, threshold={self.config.prm_pass_threshold}, passed={verification_passed}")
            
            # Record step-level training signal to ProcessRewardModel.
            # Use each step's own LATS node reward (value/visits) as a soft label
            # rather than stamping every step with the same binary verification outcome.
            # A run may have had 9 good steps and 1 bad final step; using per-node
            # rewards preserves that granularity. Falls back to verification_passed
            # when the step can't be matched to a tree node.
            from core.verification.process_reward_model import get_prm
            _prm = get_prm(self.config.embedding_model)
            step_rewards: dict = {}
            if result.lats_tree is not None:
                for node in result.lats_tree.nodes.values():
                    step_text = node.state.get("step", "")
                    if step_text and node.visits > 0:
                        step_rewards[step_text] = node.value / node.visits
            for i, step in enumerate(result.reasoning_steps):
                node_reward = step_rewards.get(step)  # float or None
                _prm.record(
                    query=query,
                    step=step,
                    context_steps=result.reasoning_steps[:i],
                    worker_type=worker_specialty,
                    verification_passed=verification_passed,
                    human_score=node_reward,  # takes priority when not None
                )
            
            if verification_passed:
                logger.info(f"PRM GATE: ✓ PASSED (avg={avg_prm:.3f})")
                logger.info(f"  - Confidence: {confidence:.3f}")
                final_result = result
                final_result.verification_passed = True
                final_result.confidence = confidence
            else:
                logger.info(f"PRM GATE: ✗ FAILED (avg={avg_prm:.3f})")
                logger.info(f"  - Confidence: {confidence:.3f}")
                if issues:
                    logger.info(f"  - Issues found ({len(issues)}):")
                    for i, issue in enumerate(issues, 1):
                        logger.info(f"    {i}. {issue}")
                
                if iteration < max_iterations:
                    # Stash tree and issues for next iteration (tree reuse)
                    previous_tree = getattr(result, 'lats_tree', None)
                    prev_issues = issues

                    logger.info("\n" + "=" * 70)
                    logger.info("STEP 5: REFLECTION")
                    logger.info("=" * 70)
                    logger.info(f"REFLECTION: Improving reasoning based on PRM gate failure")
                    logger.info(f"REFLECTION: Issues to address: {len(issues)}")
                    
                    reflection_start_time = time.time()
                    improved_steps = self.reflection_engine.enhance_reasoning(
                        query=query,
                        initial_trace=result.reasoning_steps,
                        worker_type=worker_specialty,
                        verification_issues=issues
                    )
                    reflection_time = time.time() - reflection_start_time
                    
                    logger.info(f"REFLECTION: Completed in {reflection_time:.2f}s")
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
        
        cache_data = {
            "result": response,
            "quality": "high" if verification_passed and final_result.confidence > 0.7 else "low",
            "confidence": final_result.confidence,
            "worker": worker_specialty
        }
        
        if hasattr(final_result, 'lats_tree') and final_result.lats_tree is not None:
            cache_data["lats_tree"] = final_result.lats_tree.root.to_dict()
            cache_data["tree_stats"] = {
                "total_nodes": len(final_result.lats_tree.nodes),
                "avg_reward": final_result.lats_tree.get_avg_reward(),
                "max_depth": final_result.metadata.get("tree_depth", 0)
            }
            logger.info(f"CACHE: Including full LATS tree with {len(final_result.lats_tree.nodes)} nodes")
        
        self.tree_cache.store(query_embedding, cache_data)
        logger.info(f"CACHE: Stored result with quality={cache_data['quality']}")
        
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
                "cache_quality": cache_data['quality']
            }
            self.router.record_outcome(routing_decision, outcome)
        
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
        
        if self.active_learning:
            self.active_learning.collect_query(query, response)
        
        return response
    
    def get_metrics_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        session_metrics = self.metrics.get_session_metrics(session_id)
        analytics = self.metrics.get_analytics_summary()
        
        router_metrics = {}
        feedback_metrics = {}
        if self.router:
            router_metrics = {
                "total_outcomes": len(self.router.outcomes),
                "training_buffer_size": len(self.router.training_buffer),
                "training_steps": self.router.training_step_count,
                "exploration_rate": self.router.exploration_rate
            }

            feedback_metrics = self.router.get_feedback_enhanced_stats()
        
        return {
            "session": session_metrics,
            "analytics": analytics,
            "router": router_metrics,
            "human_feedback": feedback_metrics
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
