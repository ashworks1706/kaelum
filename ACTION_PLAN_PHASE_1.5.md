# IMMEDIATE ACTION PLAN - Phase 1.5

**Start Date**: November 3, 2025  
**Duration**: 2 weeks  
**Goal**: Make routing work, foundation for workers

---

## 🎯 Week 1: Enable and Test Routing System

### Day 1 (Nov 4): Enable Routing
**Time**: 4-6 hours

**Tasks**:
1. Change default `enable_routing=False` → `True` in `kaelum/__init__.py`
2. Update all examples to use routing
3. Update README/docs to show routing
4. Test basic routing functionality

**Files to Edit**:
- `/kaelum/__init__.py` - Change default
- `/examples/*.py` - Add `enable_routing=True`
- `/README.md` - Update quickstart
- `/docs/ROUTING.md` - Mark as default feature

**Success Criteria**:
- ✅ Routing enabled by default
- ✅ All examples use routing
- ✅ Documentation updated
- ✅ Basic routing test passes

---

### Day 2 (Nov 5): Router Observability
**Time**: 6-8 hours

**Tasks**:
1. Add logging to router decisions
2. Create routing metrics collector
3. Add routing dashboard (simple CLI output)
4. Log routing decisions to file

**New Files**:
- `/kaelum/core/router_metrics.py` - Metrics collection
- `/kaelum/cli.py` - Add `kaelum routing-stats` command

**Code**:
```python
# In router.py
def route(self, query: str, context: Optional[Dict] = None) -> RoutingDecision:
    decision = self._make_decision(query, context)
    
    # Log decision
    self.logger.info(f"[ROUTING] Query: {query[:50]}")
    self.logger.info(f"[ROUTING] Type: {decision.query_type.value}")
    self.logger.info(f"[ROUTING] Strategy: {decision.strategy.value}")
    self.logger.info(f"[ROUTING] Reasoning: {decision.reasoning}")
    
    return decision
```

**Success Criteria**:
- ✅ All routing decisions logged
- ✅ Metrics collected per strategy
- ✅ CLI command shows routing stats
- ✅ Can analyze routing patterns

---

### Day 3-4 (Nov 6-7): Test on Diverse Queries
**Time**: 12-16 hours

**Tasks**:
1. Create diverse test query dataset (100+ queries)
2. Run routing on all queries
3. Analyze strategy selection accuracy
4. Identify classification errors

**New Files**:
- `/tests/test_queries.py` - Test query dataset
- `/benchmarks/routing_benchmark.py` - Routing accuracy test

**Test Categories** (20 each):
- Math: equations, word problems, calculus
- Logic: if-then, proofs, contradictions
- Code: debugging, implementation, review
- Factual: history, science, definitions
- Creative: brainstorming, writing, ideas

**Success Criteria**:
- ✅ 100+ test queries created
- ✅ All queries routed successfully
- ✅ Strategy selection accuracy measured
- ✅ Classification errors documented

---

### Day 5 (Nov 8): Improve Query Classification
**Time**: 8-10 hours

**Tasks**:
1. Analyze classification errors from Day 3-4
2. Improve regex patterns for classification
3. Add more sophisticated heuristics
4. Re-test on query dataset

**Focus Areas**:
- Multi-category queries (math + code)
- Ambiguous queries
- Complex/compound questions
- Domain-specific language

**Code Improvements**:
```python
def _classify_query(self, query: str, context: Optional[Dict]) -> QueryType:
    """Improved classification with multi-signal detection."""
    
    # Extract numerical content
    has_numbers = bool(re.findall(r'\d+', query))
    has_operators = any(op in query for op in ['+', '-', '×', '÷', '='])
    
    # Code indicators
    has_code_blocks = '```' in query or 'def ' in query or 'function' in query
    
    # Multi-signal scoring
    scores = {
        QueryType.MATH: 0,
        QueryType.CODE: 0,
        QueryType.LOGIC: 0,
        # ...
    }
    
    if has_numbers and has_operators:
        scores[QueryType.MATH] += 2
    
    # Return highest scoring type
    return max(scores.items(), key=lambda x: x[1])[0]
```

**Success Criteria**:
- ✅ Classification accuracy >85%
- ✅ Multi-category detection working
- ✅ Ambiguous queries handled better
- ✅ Re-test shows improvement

---

### Day 6-7 (Nov 9-10): Add Context Awareness
**Time**: 12-16 hours

**Tasks**:
1. Add query complexity estimation
2. Use past performance for routing
3. Add user preferences support
4. Implement adaptive thresholds

**New Features**:
```python
def route_with_context(self, query: str) -> RoutingDecision:
    """Route with context awareness."""
    
    # Estimate complexity
    complexity = self._estimate_complexity(query)
    
    # Check past performance
    similar = self.memory.find_similar(query, k=5)
    if similar:
        best_past_strategy = self._get_best_strategy(similar)
        if self._past_success_rate(best_past_strategy) > 0.9:
            return self._use_past_strategy(best_past_strategy, complexity)
    
    # Adjust for complexity
    if complexity > 0.8:
        return self._use_deep_strategy()
    elif complexity < 0.3:
        return self._use_fast_strategy()
    
    # Normal routing
    return self._route_normal(query)

def _estimate_complexity(self, query: str) -> float:
    """Estimate query complexity (0-1)."""
    factors = {
        'length': len(query.split()) / 100,  # Longer = harder
        'nesting': query.count('(') / 5,      # More nested = harder
        'multiple_parts': query.count('and') / 3,  # Multiple parts = harder
        'specific_terms': self._count_technical_terms(query) / 10
    }
    return min(1.0, sum(factors.values()))
```

**Success Criteria**:
- ✅ Complexity estimation working
- ✅ Past performance influences routing
- ✅ Simple queries use fast strategy
- ✅ Complex queries use deep strategy

---

## 🎯 Week 2: Foundation for Workers

### Day 8-9 (Nov 11-12): Design Worker Architecture
**Time**: 12-16 hours

**Tasks**:
1. Define WorkerAgent interface
2. Create BaseWorker class
3. Implement MathWorker (first worker)
4. Implement LogicWorker (second worker)
5. Test workers independently

**New Files**:
- `/kaelum/core/workers.py` - Worker architecture

**Interface Design**:
```python
class WorkerAgent(ABC):
    """Base class for specialized reasoning workers."""
    
    def __init__(self, llm: LLMClient, tools: List[Tool]):
        self.llm = llm
        self.tools = tools
        self.specialty = self._get_specialty()
    
    @abstractmethod
    def solve(self, query: str, context: Dict) -> WorkerResult:
        """Solve query using worker's specialty."""
        pass
    
    @abstractmethod
    def _get_specialty(self) -> str:
        """Return worker's specialty."""
        pass
    
    def verify(self, result: WorkerResult) -> VerificationResult:
        """Verify worker's result."""
        pass

class MathWorker(WorkerAgent):
    """Specialized in mathematical reasoning."""
    
    def _get_specialty(self) -> str:
        return "mathematical_reasoning"
    
    def solve(self, query: str, context: Dict) -> WorkerResult:
        # Use symbolic tools heavily
        trace = self.llm.generate(
            f"Solve this math problem step-by-step:\n{query}"
        )
        
        # Verify with SymPy
        verification = self.verify_symbolic(trace)
        
        return WorkerResult(
            answer=self._extract_answer(trace),
            trace=trace,
            confidence=0.95 if verification.passed else 0.6,
            verification=verification,
            worker_type="math"
        )

class LogicWorker(WorkerAgent):
    """Specialized in logical reasoning."""
    
    def _get_specialty(self) -> str:
        return "logical_reasoning"
    
    def solve(self, query: str, context: Dict) -> WorkerResult:
        # Use deep reflection
        trace = self.llm.generate(
            f"Reason through this logically:\n{query}"
        )
        
        # Multiple reflection passes
        for i in range(3):
            improved = self.reflect_on(trace)
            if self._no_improvement(trace, improved):
                break
            trace = improved
        
        return WorkerResult(
            answer=self._extract_answer(trace),
            trace=trace,
            confidence=0.85,
            worker_type="logic"
        )
```

**Success Criteria**:
- ✅ Worker interface defined
- ✅ MathWorker working
- ✅ LogicWorker working
- ✅ Workers can be tested independently

---

### Day 10-11 (Nov 13-14): Parallel Execution
**Time**: 12-16 hours

**Tasks**:
1. Add async/await support to workers
2. Implement parallel runner
3. Test with 2 workers
4. Measure speedup
5. Handle worker failures

**New Files**:
- `/kaelum/runtime/parallel_runner.py` - Parallel execution

**Implementation**:
```python
import asyncio

class ParallelRunner:
    """Run multiple workers in parallel."""
    
    def __init__(self, workers: List[WorkerAgent]):
        self.workers = {w.specialty: w for w in workers}
    
    async def run_parallel(self, query: str, worker_types: List[str]) -> List[WorkerResult]:
        """Execute multiple workers simultaneously."""
        
        tasks = []
        for worker_type in worker_types:
            if worker_type in self.workers:
                worker = self.workers[worker_type]
                task = self._run_worker_async(worker, query)
                tasks.append(task)
        
        # Run all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures
        successful = [r for r in results if isinstance(r, WorkerResult)]
        return successful
    
    async def _run_worker_async(self, worker: WorkerAgent, query: str) -> WorkerResult:
        """Run a worker asynchronously."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, worker.solve, query, {})
            return result
        except Exception as e:
            logger.error(f"Worker {worker.specialty} failed: {e}")
            raise
```

**Success Criteria**:
- ✅ Async support working
- ✅ 2 workers run in parallel
- ✅ Speedup measured (target: 1.5-2x)
- ✅ Failures handled gracefully

---

### Day 12-13 (Nov 15-16): Basic Meta-Reasoning
**Time**: 12-16 hours

**Tasks**:
1. Implement simple meta-reasoner
2. Voting strategy (consensus)
3. Confidence strategy (highest confidence)
4. Test combining 2 worker results
5. Measure quality improvement

**New Files**:
- `/kaelum/core/meta_reasoner.py` - Meta-reasoning logic

**Implementation**:
```python
class MetaReasoner:
    """Combines results from multiple workers."""
    
    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    def combine(self, results: List[WorkerResult]) -> FinalAnswer:
        """Combine worker results into final answer."""
        
        if not results:
            raise ValueError("No results to combine")
        
        # Strategy 1: Consensus (all agree)
        answers = [r.answer for r in results]
        if len(set(answers)) == 1:
            return FinalAnswer(
                answer=answers[0],
                confidence=max(r.confidence for r in results),
                method="consensus",
                worker_results=results
            )
        
        # Strategy 2: Highest confidence
        best = max(results, key=lambda r: r.confidence)
        if best.confidence > 0.9:
            return FinalAnswer(
                answer=best.answer,
                confidence=best.confidence,
                method="confidence",
                worker_results=results
            )
        
        # Strategy 3: Verification-based
        verified = [r for r in results if r.verification and r.verification.passed]
        if verified:
            best_verified = max(verified, key=lambda r: r.confidence)
            return FinalAnswer(
                answer=best_verified.answer,
                confidence=best_verified.confidence,
                method="verification",
                worker_results=results
            )
        
        # Strategy 4: Synthesize (use LLM)
        synthesis = self._synthesize_with_llm(results)
        return FinalAnswer(
            answer=synthesis,
            confidence=0.75,
            method="synthesis",
            worker_results=results
        )
    
    def _synthesize_with_llm(self, results: List[WorkerResult]) -> str:
        """Use LLM to synthesize multiple answers."""
        prompt = f"""Multiple reasoning workers provided these answers:

{self._format_results(results)}

Synthesize the best answer from these perspectives:"""
        
        return self.llm.generate(prompt)
```

**Success Criteria**:
- ✅ Meta-reasoner working
- ✅ Can combine 2 results
- ✅ All 4 strategies working
- ✅ Quality improvement measured (target: >10%)

---

### Day 14 (Nov 17): Integration and Testing
**Time**: 8-10 hours

**Tasks**:
1. Integrate workers into orchestrator
2. Add routing → workers → meta-reasoning pipeline
3. End-to-end testing
4. Benchmark vs single strategy
5. Document improvements

**Integration**:
```python
class KaelumOrchestrator:
    def __init__(self, ..., enable_workers: bool = False):
        # ... existing code ...
        
        if enable_workers:
            self.workers = {
                "math": MathWorker(self.llm, [sympy_tool]),
                "logic": LogicWorker(self.llm, [])
            }
            self.parallel_runner = ParallelRunner(list(self.workers.values()))
            self.meta_reasoner = MetaReasoner(self.llm)
    
    def infer_with_workers(self, query: str) -> Dict[str, Any]:
        """Infer using multiple workers."""
        
        # Route to determine worker types
        decision = self.router.route(query)
        worker_types = self._select_workers(decision.query_type)
        
        # Run workers in parallel
        worker_results = asyncio.run(
            self.parallel_runner.run_parallel(query, worker_types)
        )
        
        # Meta-reason to combine
        final = self.meta_reasoner.combine(worker_results)
        
        return final
```

**Success Criteria**:
- ✅ Full pipeline working
- ✅ Routing → Workers → Meta-reasoning
- ✅ Benchmarks show improvement
- ✅ Documentation complete

---

## 📊 Success Metrics (Week 1-2)

### Week 1 Targets
- ✅ Routing enabled and tested
- ✅ 100+ test queries analyzed
- ✅ Classification accuracy >85%
- ✅ Context awareness working
- ✅ Router observability dashboard

### Week 2 Targets
- ✅ 2 workers implemented (Math, Logic)
- ✅ Parallel execution working
- ✅ 1.5-2x speedup from parallel
- ✅ Meta-reasoner combining results
- ✅ >10% quality improvement

### Overall Phase 1.5 Success
- ✅ Routing is DEFAULT and WORKING
- ✅ Foundation for mixture of experts ready
- ✅ Parallel execution proven
- ✅ Quality improvements measured
- ✅ Ready for Phase 2 (more workers, learning)

---

## 🎯 Next Steps After Phase 1.5

With this foundation, we can:
1. Add 3 more workers (Code, Factual, Creative)
2. Implement controller model (neural routing)
3. Add reasoning memory (learning)
4. Improve meta-reasoning strategies
5. Scale to 5+ workers running in parallel

**Timeline**: Phase 2 starts Nov 18, 2025

---

**Let's make routing actually work!** 🚀
