"""Core orchestration logic for Kaelum reasoning system.

New Architecture:
Query → Router → Worker → LATS → Result

Workers use LATS (Language Agent Tree Search) for multi-step reasoning with:
- MCTS-style exploration
- Tree caching for similar queries
- Domain-specific verification
"""

import time
import logging
from typing import Iterator, Dict, Any, Optional

from ..core.config import KaelumConfig
from ..core.reasoning import LLMClient
from ..core.metrics import CostTracker
from ..core.router import Router
from ..core.workers import WorkerSpecialty, create_worker
from ..core.tree_cache import TreeCache

try:
    from ..core.neural_router import NeuralRouter
    NEURAL_ROUTER_AVAILABLE = True
except ImportError:
    NEURAL_ROUTER_AVAILABLE = False
    NeuralRouter = None

logger = logging.getLogger("kaelum.orchestrator")


class KaelumOrchestrator:
    """Orchestrates reasoning pipeline: Router → Worker → LATS → Answer"""

    def __init__(self, config: KaelumConfig, rag_adapter=None, reasoning_system_prompt=None, 
                 reasoning_user_template=None, enable_routing: bool = True, use_neural_router: bool = False):
        self.config = config
        self.llm = LLMClient(config.reasoning_llm)
        self.metrics = CostTracker()
        self.tree_cache = TreeCache()
        
        # Router for worker selection
        self.router = None
        self.enable_routing = enable_routing
        
        if enable_routing:
            if use_neural_router and NEURAL_ROUTER_AVAILABLE:
                try:
                    self.router = NeuralRouter(fallback_to_rules=True)
                    logger.info("Using Neural Router for worker selection")
                except Exception as e:
                    logger.warning(f"Neural Router failed to initialize: {e}")
                    logger.info("Falling back to rule-based router")
                    self.router = Router(learning_enabled=True)
            else:
                self.router = Router(learning_enabled=True)
                logger.info("Using rule-based router for worker selection")
        
        # Worker cache
        self._workers = {}
        self.rag_adapter = rag_adapter

    def _get_worker(self, specialty: str):
        """Get or create worker for given specialty."""
        if specialty not in self._workers:
            try:
                specialty_enum = WorkerSpecialty(specialty)
                if specialty == "factual":
                    self._workers[specialty] = create_worker(
                        specialty_enum, self.config, 
                        rag_adapter=self.rag_adapter
                    )
                else:
                    self._workers[specialty] = create_worker(specialty_enum, self.config)
                self._workers[specialty].tree_cache = self.tree_cache
            except Exception as e:
                logger.error(f"Failed to create {specialty} worker: {e}")
                # Fallback to logic worker
                self._workers[specialty] = create_worker(WorkerSpecialty.LOGIC, self.config)
                self._workers[specialty].tree_cache = self.tree_cache
        
        return self._workers[specialty]
    
    def infer(self, query: str, stream: bool = False):
        """Run reasoning pipeline with worker-based routing and LATS."""
        if stream:
            logger.warning("Streaming not yet supported in worker-based mode, using sync")
            stream = False
        
        # Route query to appropriate worker
        if self.enable_routing and self.router:
            routing_decision = self.router.route(query)
            worker_specialty = routing_decision.worker_specialty
            use_cache = routing_decision.use_tree_cache
            max_depth = routing_decision.max_tree_depth
            num_sims = routing_decision.num_simulations
        else:
            # Default routing
            worker_specialty = "logic"
            use_cache = True
            max_depth = 5
            num_sims = 10
        
        # Get worker
        worker = self._get_worker(worker_specialty)
        
        # Execute worker with LATS
        session_id = f"worker_{int(time.time() * 1000)}"
        self.metrics.start_session(session_id, metadata={"query": query[:50]})
        start_time = time.time()
        
        logger.info(f"Executing {worker_specialty} worker with LATS")
        logger.info(f"LATS config: depth={max_depth}, simulations={num_sims}, cache={use_cache}")
        
        result = worker.solve(query, context=None, use_cache=use_cache,
                            max_tree_depth=max_depth, num_simulations=num_sims)
        
        total_time = (time.time() - start_time) * 1000
        
        # Record outcome for router learning
        if self.enable_routing and self.router:
            outcome = {
                "query": query,
                "success": result.verification_passed,
                "confidence": result.confidence,
                "execution_time": result.execution_time,
                "cost": result.execution_time * 0.00000001  # Estimate
            }
            self.router.record_outcome(routing_decision, outcome)
        
        # Format response
        return {
            "query": query,
            "reasoning_trace": result.reasoning_steps,
            "answer": result.answer,
            "worker": worker_specialty,
            "confidence": result.confidence,
            "verification_passed": result.verification_passed,
            "cache_hit": result.metadata.get("cache_hit", False),
            "metrics": {
                "total_time_ms": total_time,
                "execution_time_ms": result.execution_time * 1000,
                "tree_depth": result.metadata.get("tree_depth", 0),
                "num_simulations": result.metadata.get("num_simulations", 0)
            }
        }
