"""Core orchestration logic for Kaelum reasoning system."""

import time
from typing import Iterator, Dict, Any, Optional

from ..core.config import KaelumConfig
from ..core.reasoning import ReasoningGenerator, LLMClient
from ..core.verification import VerificationEngine
from ..core.reflection import ReflectionEngine
from ..core.metrics import CostTracker
from ..core.router import Router

try:
    from ..core.neural_router import NeuralRouter
    NEURAL_ROUTER_AVAILABLE = True
except ImportError:
    NEURAL_ROUTER_AVAILABLE = False
    NeuralRouter = None


class KaelumOrchestrator:
    """Orchestrates reasoning pipeline: Generate → Verify → Reflect → Answer"""

    def __init__(self, config: KaelumConfig, rag_adapter=None, reasoning_system_prompt=None, 
                 reasoning_user_template=None, enable_routing: bool = False, use_neural_router: bool = True):
        self.config = config
        self.llm = LLMClient(config.reasoning_llm)
        self.generator = ReasoningGenerator(
            self.llm,
            system_prompt=reasoning_system_prompt,
            user_template=reasoning_user_template
        )
        self.reflection = ReflectionEngine(self.llm, max_iterations=config.max_reflection_iterations)
        self.verification = VerificationEngine(
            use_symbolic=config.use_symbolic_verification,
            use_factual_check=config.use_factual_verification,
            rag_adapter=rag_adapter,
            debug=config.debug_verification
        )
        self.metrics = CostTracker()
        
        # Router for adaptive strategy selection (Phase 2 feature)
        # Try to use neural router if available and requested
        self.router = None
        self.enable_routing = enable_routing
        
        if enable_routing:
            if use_neural_router and NEURAL_ROUTER_AVAILABLE:
                try:
                    self.router = NeuralRouter(fallback_to_rules=True)
                    print("✓ Using Neural Router (Kaelum Brain)")
                except Exception as e:
                    print(f"⚠️  Neural Router failed to initialize: {e}")
                    print("   Falling back to rule-based router")
                    self.router = Router(learning_enabled=True)
            else:
                self.router = Router(learning_enabled=True)
                print("✓ Using rule-based router")

    def infer(self, query: str, stream: bool = False):
        """Run reasoning pipeline with optional adaptive routing."""
        # If routing is enabled, get optimal strategy for this query
        if self.enable_routing and self.router:
            routing_decision = self.router.route(query)
            
            # Temporarily override config with routed strategy
            original_config = {
                "max_reflection_iterations": self.config.max_reflection_iterations,
                "use_symbolic_verification": self.verification.symbolic_verifier is not None,
                "use_factual_verification": self.verification.factual_verifier is not None
            }
            
            # Apply routing decision
            self.config.max_reflection_iterations = routing_decision.max_reflection_iterations
            self.reflection.max_iterations = routing_decision.max_reflection_iterations
            
            # Run inference
            result = self._infer_stream(query) if stream else self._infer_sync(query, routing_decision)
            
            # Record outcome for learning (only for sync mode)
            if not stream and self.router:
                self.router.record_outcome(routing_decision, result)
            
            # Restore original config
            self.config.max_reflection_iterations = original_config["max_reflection_iterations"]
            self.reflection.max_iterations = original_config["max_reflection_iterations"]
            
            return result
        else:
            # Standard inference without routing
            return self._infer_stream(query) if stream else self._infer_sync(query, None)
    
    def _infer_sync(self, query: str, routing_decision=None) -> Dict[str, Any]:
        """Synchronous inference with full verification + reflection."""
        session_id = f"sync_{int(time.time() * 1000)}"
        self.metrics.start_session(session_id, metadata={"query": query[:50]})
        start_time = time.time()
        
        # Step 1: Generate reasoning trace
        trace, reasoning_time = self._generate_trace_sync(query, session_id)
        
        # Step 2: Verify reasoning
        errors, details, verify_time = self._verify_trace_timed(trace)
        
        # Step 3: Reflect if needed
        trace, reflection_time = self._reflect_if_needed(query, trace, errors, session_id)
        
        # Step 4: Generate final answer
        answer, answer_time = self._generate_answer_sync(query, trace, session_id)
        
        # Calculate and return metrics
        return self._build_result(
            query, trace, answer, errors, details,
            start_time, reasoning_time, verify_time, reflection_time, answer_time,
            session_id
        )
    
    def _generate_trace_sync(self, query: str, session_id: str) -> tuple:
        """Generate reasoning trace and return parsed steps with timing."""
        reasoning_start = time.time()
        trace_text = self.generator.generate_reasoning(query, stream=False)
        reasoning_time = (time.time() - reasoning_start) * 1000
        reasoning_tokens = len(trace_text.split())
        
        self.metrics.log_inference(
            model_type="local_reasoning",
            tokens=reasoning_tokens,
            latency_ms=reasoning_time,
            cost=reasoning_tokens * 0.00000001,
            session_id=session_id
        )
        
        trace = self._parse_trace(trace_text)
        return trace, reasoning_time
    
    def _parse_trace(self, trace_text: str) -> list:
        """Parse reasoning trace text into list of steps."""
        trace = []
        for line in trace_text.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                step = line.lstrip("0123456789.-•) ").strip()
                if step:
                    trace.append(step)
        return trace if trace else [trace_text.strip()]
    
    def _verify_trace_timed(self, trace: list) -> tuple:
        """Verify trace and return errors, details, and timing."""
        verify_start = time.time()
        errors, details = self.verification.verify_trace(trace)
        verify_time = (time.time() - verify_start) * 1000
        return errors, details, verify_time
    
    def _reflect_if_needed(self, query: str, trace: list, errors: list, session_id: str) -> tuple:
        """Apply reflection if needed and return enhanced trace with timing."""
        reflection_time = 0
        if errors or self.config.max_reflection_iterations > 0:
            reflect_start = time.time()
            trace = self.reflection.enhance_reasoning(query, trace)
            reflection_time = (time.time() - reflect_start) * 1000
            
            reflection_tokens = sum(len(step.split()) for step in trace)
            self.metrics.log_inference(
                model_type="local_reflection",
                tokens=reflection_tokens,
                latency_ms=reflection_time,
                cost=reflection_tokens * 0.00000001,
                session_id=session_id
            )
        return trace, reflection_time
    
    def _generate_answer_sync(self, query: str, trace: list, session_id: str) -> tuple:
        """Generate final answer and return text with timing."""
        answer_start = time.time()
        answer = self.generator.generate_answer(query, trace, stream=False)
        answer_time = (time.time() - answer_start) * 1000
        answer_tokens = len(answer.split())
        
        self.metrics.log_inference(
            model_type="local_answer",
            tokens=answer_tokens,
            latency_ms=answer_time,
            cost=answer_tokens * 0.00000001,
            session_id=session_id
        )
        return answer, answer_time
    
    def _build_result(self, query: str, trace: list, answer: str, errors: list, details: dict,
                     start_time: float, reasoning_time: float, verify_time: float, 
                     reflection_time: float, answer_time: float, session_id: str) -> dict:
        """Build final result dictionary with all metrics."""
        total_time = (time.time() - start_time) * 1000
        session_metrics = self.metrics.get_session_metrics(session_id)
        savings = self.metrics.calculate_savings(session_id)
        
        return {
            "query": query,
            "reasoning_trace": trace,
            "answer": answer,
            "verification_errors": errors,
            "verification_details": details,
            "metrics": {
                "total_time_ms": total_time,
                "reasoning_time_ms": reasoning_time,
                "verification_time_ms": verify_time,
                "reflection_time_ms": reflection_time,
                "answer_time_ms": answer_time,
                "total_tokens": session_metrics['total_tokens'],
                "local_cost": session_metrics['total_cost'],
                "commercial_cost": savings['commercial_cost'],
                "savings": savings['savings'],
                "savings_percent": savings['savings_percent']
            }
        }
    
    def _infer_stream(self, query: str):
        """Streaming inference with full verification + reflection."""
        session_id = f"stream_{int(time.time() * 1000)}"
        self.metrics.start_session(session_id, metadata={"query": query[:50]})
        start_time = time.time()
        
        # Step 1: Stream reasoning trace
        yield "🧠 [REASON]\n\n"
        trace, reasoning_time = yield from self._stream_reasoning(query, session_id)
        
        # Step 2: Stream verification
        errors, details, verify_time = yield from self._stream_verification(trace)
        
        # Step 3: Stream reflection if needed
        trace, reflection_time = yield from self._stream_reflection(query, trace, errors, session_id)
        
        # Step 4: Stream final answer
        answer_time = yield from self._stream_answer(query, trace, session_id)
        
        # Step 5: Stream metrics summary
        yield from self._stream_metrics(start_time, reasoning_time, verify_time, 
                                       reflection_time, answer_time, session_id)
    
    def _stream_reasoning(self, query: str, session_id: str):
        """Stream reasoning generation and return trace with timing."""
        reasoning_start = time.time()
        trace_text = ""
        for chunk in self.generator.generate_reasoning(query, stream=True):
            trace_text += chunk
            yield chunk
        
        reasoning_time = (time.time() - reasoning_start) * 1000
        reasoning_tokens = len(trace_text.split())
        
        self.metrics.log_inference(
            model_type="local_reasoning",
            tokens=reasoning_tokens,
            latency_ms=reasoning_time,
            cost=reasoning_tokens * 0.00000001,
            session_id=session_id
        )
        
        trace = self._parse_trace(trace_text)
        return trace, reasoning_time
    
    def _stream_verification(self, trace: list):
        """Stream verification results and return errors, details, timing."""
        yield "\n\n🔍 [VERIFY]\n"
        verify_start = time.time()
        errors, details = self.verification.verify_trace(trace)
        verify_time = (time.time() - verify_start) * 1000
        
        yield f"   {details['total_steps']} steps"
        
        if self.verification.symbolic_verifier:
            yield f" | Symbolic: {details['symbolic_passed']}/{details['symbolic_checks']}"
        
        if self.verification.factual_verifier:
            yield f" | Factual: {details['factual_passed']}/{details['factual_checks']}"
        
        yield f" | {verify_time:.1f}ms\n"
        
        if errors:
            yield f"   ⚠️ {len(errors)} issue(s):\n"
            for error in errors:
                yield f"      - {error}\n"
        
        return errors, details, verify_time
    
    def _stream_reflection(self, query: str, trace: list, errors: list, session_id: str):
        """Stream reflection process and return enhanced trace with timing."""
        reflection_time = 0
        if errors or self.config.max_reflection_iterations > 0:
            yield f"\n🔄 [REFLECT]\n"
            
            reflect_start = time.time()
            
            for i in range(self.config.max_reflection_iterations):
                issues = self.reflection._verify_trace(query, trace)
                if not issues:
                    break
                
                if i < self.config.max_reflection_iterations - 1:
                    trace = self.reflection._improve_trace(query, trace, issues)
                    for j, step in enumerate(trace):
                        if j == 0:
                            yield f"\n"
                        yield f"   {j+1}. {step}\n"
                    yield f"\n"
            
            reflection_time = (time.time() - reflect_start) * 1000
            
            reflection_tokens = sum(len(step.split()) for step in trace)
            self.metrics.log_inference(
                model_type="local_reflection",
                tokens=reflection_tokens,
                latency_ms=reflection_time,
                cost=reflection_tokens * 0.00000001,
                session_id=session_id
            )
        
        return trace, reflection_time
    
    def _stream_answer(self, query: str, trace: list, session_id: str):
        """Stream final answer generation and return timing."""
        yield "\n✅ [ANSWER]\n\n"
        answer_start = time.time()
        answer_text = ""
        for chunk in self.generator.generate_answer(query, trace, stream=True):
            answer_text += chunk
            yield chunk
        
        answer_time = (time.time() - answer_start) * 1000
        answer_tokens = len(answer_text.split())
        
        self.metrics.log_inference(
            model_type="local_answer",
            tokens=answer_tokens,
            latency_ms=answer_time,
            cost=answer_tokens * 0.00000001,
            session_id=session_id
        )
        
        return answer_time
    
    def _stream_metrics(self, start_time: float, reasoning_time: float, verify_time: float,
                       reflection_time: float, answer_time: float, session_id: str):
        """Stream final metrics summary."""
        total_time = (time.time() - start_time) * 1000
        session_metrics = self.metrics.get_session_metrics(session_id)
        savings = self.metrics.calculate_savings(session_id)
        
        yield f"\n\n{'='*70}\n"
        yield f"📊 [METRICS]\n"
        yield f"   Total: {total_time:.1f}ms"
        yield f" | Reason: {reasoning_time:.1f}ms"
        yield f" | Verify: {verify_time:.1f}ms"
        if reflection_time > 0:
            yield f" | Reflect: {reflection_time:.1f}ms"
        yield f" | Answer: {answer_time:.1f}ms\n"
        yield f"   Tokens: {session_metrics['total_tokens']}"
        yield f" | Cost: ${session_metrics['total_cost']:.8f}"
        yield f" | vs Commercial: ${savings['commercial_cost']:.4f}"
        yield f" | 💰 {savings['savings_percent']:.1f}% savings\n"
        yield f"{'='*70}\n"
