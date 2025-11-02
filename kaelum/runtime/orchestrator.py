"""Core orchestration logic for Kaelum reasoning system."""

import time
from typing import Iterator, Dict, Any

from ..core.config import KaelumConfig
from ..core.reasoning import ReasoningGenerator, LLMClient
from ..core.verification import VerificationEngine
from ..core.reflection import ReflectionEngine
from ..core.metrics import CostTracker


class KaelumOrchestrator:
    """Orchestrates reasoning pipeline: Generate → Verify → Reflect → Answer"""

    def __init__(self, config: KaelumConfig, rag_adapter=None, reasoning_system_prompt=None, reasoning_user_template=None):
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
            rag_adapter=rag_adapter
        )
        self.metrics = CostTracker()

    def infer(self, query: str, stream: bool = False):
        """Run reasoning pipeline."""
        if stream:
            return self._infer_stream(query)
        else:
            return self._infer_sync(query)
    
    def _infer_sync(self, query: str) -> Dict[str, Any]:
        """Synchronous inference with full verification + reflection."""
        session_id = f"sync_{int(time.time() * 1000)}"
        self.metrics.start_session(session_id, metadata={"query": query[:50]})
        
        start_time = time.time()
        
        # Step 1: Generate reasoning trace
        reasoning_start = time.time()
        trace_text = self.generator.generate_reasoning(query, stream=False)
        reasoning_time = (time.time() - reasoning_start) * 1000
        reasoning_tokens = len(trace_text.split())
        
        # Log reasoning generation
        self.metrics.log_inference(
            model_type="local_reasoning",
            tokens=reasoning_tokens,
            latency_ms=reasoning_time,
            cost=reasoning_tokens * 0.00000001,
            session_id=session_id
        )
        
        # Parse reasoning steps
        trace = []
        for line in trace_text.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                step = line.lstrip("0123456789.-•) ").strip()
                if step:
                    trace.append(step)
        
        if not trace:
            trace = [trace_text.strip()]
        
        # Step 2: Verify reasoning
        verify_start = time.time()
        errors, details = self.verification.verify_trace(trace)
        verify_time = (time.time() - verify_start) * 1000
        
        # Step 3: Reflect if needed
        reflection_time = 0
        if errors or self.config.max_reflection_iterations > 0:
            reflect_start = time.time()
            trace = self.reflection.enhance_reasoning(query, trace)
            reflection_time = (time.time() - reflect_start) * 1000
            
            # Log reflection
            reflection_tokens = sum(len(step.split()) for step in trace)
            self.metrics.log_inference(
                model_type="local_reflection",
                tokens=reflection_tokens,
                latency_ms=reflection_time,
                cost=reflection_tokens * 0.00000001,
                session_id=session_id
            )
        
        # Step 4: Generate final answer
        answer_start = time.time()
        answer = self.generator.generate_answer(query, trace, stream=False)
        answer_time = (time.time() - answer_start) * 1000
        answer_tokens = len(answer.split())
        
        # Log answer generation
        self.metrics.log_inference(
            model_type="local_answer",
            tokens=answer_tokens,
            latency_ms=answer_time,
            cost=answer_tokens * 0.00000001,
            session_id=session_id
        )
        
        # Calculate metrics
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
        
        yield "🧠 [Reasoning]...\n\n"
        
        # Step 1: Generate reasoning trace (streaming)
        reasoning_start = time.time()
        trace_text = ""
        for chunk in self.generator.generate_reasoning(query, stream=True):
            trace_text += chunk
            yield chunk
        
        reasoning_time = (time.time() - reasoning_start) * 1000
        reasoning_tokens = len(trace_text.split())
        
        # Log reasoning generation
        self.metrics.log_inference(
            model_type="local_reasoning",
            tokens=reasoning_tokens,
            latency_ms=reasoning_time,
            cost=reasoning_tokens * 0.00000001,  # Local model cost
            session_id=session_id
        )
        
        # Parse reasoning steps
        trace = []
        for line in trace_text.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                step = line.lstrip("0123456789.-•) ").strip()
                if step:
                    trace.append(step)
        
        if not trace:
            trace = [trace_text.strip()]
        
        # Step 2: Verify reasoning
        yield "\n\n🔍 [Verification]\n"
        verify_start = time.time()
        errors, details = self.verification.verify_trace(trace)
        verify_time = (time.time() - verify_start) * 1000
        
        yield f"   Steps analyzed: {details['total_steps']}\n"
        
        if self.verification.symbolic_verifier:
            yield f"   Symbolic checks: {details['symbolic_passed']}/{details['symbolic_checks']} passed\n"
        
        if self.verification.factual_verifier:
            yield f"   Factual checks: {details['factual_passed']}/{details['factual_checks']} passed\n"
        
        yield f"   Verification time: {verify_time:.1f}ms\n"
        
        if errors:
            yield f"\n   ⚠️ Issues found:\n"
            for error in errors:
                yield f"      - {error}\n"
        else:
            yield f"   ✓ All verification checks passed\n"
        
        # Step 3: Reflect if needed
        reflection_time = 0
        if errors or self.config.max_reflection_iterations > 0:
            yield f"\n🔄 [Reflection]\n"
            yield f"   Max iterations: {self.config.max_reflection_iterations}\n"
            
            if errors:
                yield f"   Reason: Found {len(errors)} error(s) to fix\n"
            else:
                yield f"   Reason: Performing quality enhancement\n"
            
            yield f"[Reasoning]\n"
            reflect_start = time.time()
            trace = self.reflection.enhance_reasoning(query, trace)
            reflection_time = (time.time() - reflect_start) * 1000
            
            # Log reflection
            reflection_tokens = sum(len(step.split()) for step in trace)
            self.metrics.log_inference(
                model_type="local_reflection",
                tokens=reflection_tokens,
                latency_ms=reflection_time,
                cost=reflection_tokens * 0.00000001,
                session_id=session_id
            )
            
            yield f"   Reflection time: {reflection_time:.1f}ms\n"
            yield f"   ✓ Reasoning improved\n"
        
        # Step 4: Generate final answer (streaming)
        yield "\n✅ [Final Answer]\n\n"
        answer_start = time.time()
        answer_text = ""
        for chunk in self.generator.generate_answer(query, trace, stream=True):
            answer_text += chunk
            yield chunk
        
        answer_time = (time.time() - answer_start) * 1000
        answer_tokens = len(answer_text.split())
        
        # Log answer generation
        self.metrics.log_inference(
            model_type="local_answer",
            tokens=answer_tokens,
            latency_ms=answer_time,
            cost=answer_tokens * 0.00000001,
            session_id=session_id
        )
        
        # Final metrics summary
        total_time = (time.time() - start_time) * 1000
        session_metrics = self.metrics.get_session_metrics(session_id)
        savings = self.metrics.calculate_savings(session_id)
        
        yield f"\n\n{'='*70}\n"
        yield f"📊 [Metrics]\n"
        yield f"   Total time: {total_time:.1f}ms\n"
        yield f"   - Reasoning: {reasoning_time:.1f}ms\n"
        yield f"   - Verification: {verify_time:.1f}ms\n"
        if reflection_time > 0:
            yield f"   - Reflection: {reflection_time:.1f}ms\n"
        yield f"   - Answer: {answer_time:.1f}ms\n"
        yield f"\n"
        yield f"   Total tokens: {session_metrics['total_tokens']}\n"
        yield f"   Local cost: ${session_metrics['total_cost']:.8f}\n"
        yield f"   Commercial cost (est): ${savings['commercial_cost']:.4f}\n"
        yield f"   💰 Savings: ${savings['savings']:.4f} ({savings['savings_percent']:.1f}%)\n"
        yield f"{'='*70}\n"
