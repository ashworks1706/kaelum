# 🔥 MAJOR REFACTOR: Building Real Innovation

**Date**: November 4, 2025  
**Goal**: Transform from pattern-matching test-passing system to genuinely innovative AI reasoning platform

---

## 🎯 Core Principles

1. **NO MOCKS** - All tests use real LLM, fail if not running
2. **NO PATTERN MATCHING** - Replace regex with embeddings and actual learning
3. **NO SIMPLE VOTING** - Workers debate, critique, and refine through iteration
4. **REAL SPECIALIZATION** - Workers use different reasoning methods, not just different prompts

---

## 🚀 Phase 1: Infrastructure Cleanup (NOW)

### LLM Requirements
- **Model**: Qwen/Qwen2.5-3B-Instruct (smaller, faster for iteration)
- **Alternative**: Qwen/Qwen2.5-1.5B-Instruct (even faster)
- **Server**: vLLM, Ollama, or LM Studio
- **Endpoint**: `http://localhost:8000/v1`
- **Status Check**: Tests MUST fail with clear message if LLM offline

### Remove All Mocks
- [ ] Delete `tests/conftest.py` mock fixtures
- [ ] Remove all `unittest.mock` imports from tests
- [ ] Create real LLM client fixture that checks connection
- [ ] Update all test files to use real LLM
- [ ] Accept that tests will be slower - that's reality

### Test Infrastructure
```python
@pytest.fixture(scope="session")
def real_llm_client():
    """Real LLM client - tests fail if not available."""
    config = KaelumConfig()
    client = LLMClient(config.reasoning_llm)
    
    # Check if LLM is actually running
    try:
        test_response = client.generate([Message(role="user", content="Test")])
        if not test_response:
            raise ConnectionError("LLM returned empty response")
    except Exception as e:
        pytest.fail(
            f"❌ LLM not available at {config.reasoning_llm.base_url}\n"
            f"Error: {e}\n\n"
            f"Start LLM with:\n"
            f"  vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000\n"
            f"  OR\n"
            f"  ollama run qwen2.5:3b\n"
        )
    
    return client
```

---

## 🧠 Phase 2: Intelligent Router (Replace Keyword Matching)

### Current Problem
```python
# This is NOT intelligent:
if "calculate" in query or "solve" in query:
    return QueryType.MATH
```

### Solution: Embedding-Based Classification

**Use sentence-transformers for query understanding:**

```python
class IntelligentRouter:
    def __init__(self):
        # Load lightweight embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB
        
        # Load query examples with embeddings
        self.query_examples = self._load_training_data()
        self.example_embeddings = self.encoder.encode(
            [ex['query'] for ex in self.query_examples]
        )
    
    def classify(self, query: str) -> QueryType:
        # Encode query
        query_embedding = self.encoder.encode([query])[0]
        
        # Find K nearest neighbors
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.example_embeddings
        )[0]
        
        # Get top-K most similar examples
        top_k = np.argsort(similarities)[-5:]
        
        # Weighted vote from neighbors
        type_scores = defaultdict(float)
        for idx in top_k:
            example = self.query_examples[idx]
            type_scores[example['type']] += similarities[idx]
        
        return max(type_scores.items(), key=lambda x: x[1])[0]
```

### Learning Loop
```python
def learn_from_outcome(self, query: str, predicted_type: QueryType, 
                      actual_performance: Dict[str, float]):
    """
    Add successful queries to training data.
    Retrain embeddings periodically.
    """
    if actual_performance['accuracy'] > 0.8:
        self.query_examples.append({
            'query': query,
            'type': predicted_type,
            'performance': actual_performance
        })
        
        # Retrain every 100 examples
        if len(self.query_examples) % 100 == 0:
            self._retrain_classifier()
```

### Small Classifier Model (Advanced)
```python
# Phase 2b: Train Qwen2.5-1.5B as router
class NeuralRouter:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            num_labels=6  # 6 query types
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    def classify(self, query: str) -> Tuple[QueryType, float]:
        inputs = self.tokenizer(query, return_tensors="pt")
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        
        predicted_idx = torch.argmax(probs).item()
        confidence = probs[predicted_idx].item()
        
        return QueryType(predicted_idx), confidence
```

---

## 🎯 Phase 3: Specialized Workers (Real Intelligence)

### Current Problem
Workers are just LLM wrappers with different system prompts.

### Solution: Different Reasoning Methods

#### MathWorker - Keep SymPy Excellence
```python
class MathWorker:
    def solve(self, query: str) -> WorkerResult:
        # KEEP: SymPy symbolic solving (this IS innovative)
        sympy_result = self.sympy_engine.solve(query)
        
        # NEW: If SymPy fails, use chain-of-thought with verification
        if not sympy_result:
            llm_result = self._solve_with_cot(query)
            # Verify each step symbolically
            verified_steps = self._verify_steps_symbolically(llm_result.steps)
            return verified_steps
        
        return sympy_result
```

#### LogicWorker - Formal Logic + LLM
```python
class LogicWorker:
    def solve(self, query: str) -> WorkerResult:
        # 1. Extract logical structure
        predicates = self._extract_predicates(query)
        rules = self._extract_rules(query)
        
        # 2. Try formal logic solver (Z3, PyDatalog)
        formal_solution = self._solve_formally(predicates, rules)
        
        # 3. LLM provides natural language reasoning
        llm_explanation = self._get_llm_explanation(query, formal_solution)
        
        # 4. Verify consistency between formal and natural language
        return self._verify_consistency(formal_solution, llm_explanation)
```

#### CodeWorker - Execute + Analyze
```python
class CodeWorker:
    def solve(self, query: str) -> WorkerResult:
        # 1. Generate code
        code = self._generate_code(query)
        
        # 2. ACTUALLY EXECUTE in sandbox
        execution_result = self._execute_in_sandbox(code)
        
        # 3. Analyze execution
        if execution_result.failed:
            # Debug and fix
            fixed_code = self._debug_code(code, execution_result.error)
            execution_result = self._execute_in_sandbox(fixed_code)
        
        # 4. Verify output matches expected behavior
        return self._verify_output(execution_result, query)
```

#### FactualWorker - RAG + Multi-Source Verification
```python
class FactualWorker:
    def solve(self, query: str) -> WorkerResult:
        # 1. Retrieve from multiple sources
        wiki_results = self.wiki_rag.retrieve(query)
        web_results = self.web_rag.retrieve(query)
        
        # 2. Cross-reference sources
        verified_facts = self._cross_reference(wiki_results, web_results)
        
        # 3. LLM synthesizes with citations
        answer = self._synthesize_with_citations(query, verified_facts)
        
        # 4. Fact-check answer against sources
        fact_check = self._fact_check(answer, verified_facts)
        
        return WorkerResult(
            answer=answer,
            confidence=fact_check.confidence,
            sources=verified_facts,
            verification_passed=fact_check.all_verified
        )
```

#### CreativeWorker - Divergent Thinking
```python
class CreativeWorker:
    def solve(self, query: str) -> WorkerResult:
        # 1. Generate MULTIPLE diverse solutions (temperature=0.9)
        solutions = []
        for _ in range(5):
            solution = self._generate_creative(query, temperature=0.9)
            solutions.append(solution)
        
        # 2. Evaluate diversity and novelty
        diversity_score = self._calculate_diversity(solutions)
        
        # 3. Select best balance of novelty + quality
        best_solution = self._select_best(solutions, diversity_score)
        
        return WorkerResult(
            answer=best_solution,
            confidence=self._assess_quality(best_solution),
            metadata={'diversity': diversity_score, 'alternatives': len(solutions)}
        )
```

---

## 🤖 Phase 4: Intelligent MetaReasoner (Worker Debate)

### Current Problem
Simple voting/confidence picking. No actual intelligence in combination.

### Solution: Iterative Debate & Refinement

```python
class SmartMetaReasoner:
    async def reason_with_debate(self, query: str, max_rounds: int = 3) -> MetaResult:
        """Workers debate and refine answers through multiple rounds."""
        
        # Round 1: Initial solutions
        solutions = await self._get_initial_solutions(query)
        
        for round_num in range(max_rounds):
            # Each worker critiques others' solutions
            critiques = await self._get_cross_critiques(solutions)
            
            # Workers refine based on critiques
            refined_solutions = await self._refine_solutions(solutions, critiques)
            
            # Check for convergence
            if self._has_converged(solutions, refined_solutions):
                break
            
            solutions = refined_solutions
        
        # Final synthesis with reasoning about disagreements
        final_answer = await self._synthesize_with_reasoning(solutions)
        
        return MetaResult(
            answer=final_answer.answer,
            confidence=final_answer.confidence,
            debate_rounds=round_num + 1,
            worker_positions=solutions,
            synthesis_reasoning=final_answer.reasoning
        )
    
    async def _get_cross_critiques(self, solutions: List[WorkerResult]) -> List[Critique]:
        """Each worker critiques others' solutions."""
        critiques = []
        
        for worker in self.workers:
            # Show worker all OTHER workers' solutions
            other_solutions = [s for s in solutions if s.worker != worker]
            
            critique = await worker.critique_others(
                query=self.current_query,
                other_solutions=other_solutions,
                own_solution=self._find_own_solution(solutions, worker)
            )
            
            critiques.append(critique)
        
        return critiques
    
    async def _synthesize_with_reasoning(self, solutions: List[WorkerResult]) -> FinalAnswer:
        """Synthesize final answer with explicit reasoning about disagreements."""
        
        # Identify points of agreement and disagreement
        agreements = self._find_agreements(solutions)
        disagreements = self._find_disagreements(solutions)
        
        # Use LLM to reason about WHY workers disagree
        disagreement_analysis = await self.llm_client.generate([
            Message(role="system", content=
                "You are a meta-reasoner analyzing why different AI workers disagree. "
                "Explain the root cause of disagreements and determine which perspective is more valid."
            ),
            Message(role="user", content=
                f"Query: {self.current_query}\n\n"
                f"Agreements: {agreements}\n\n"
                f"Disagreements: {disagreements}\n\n"
                f"Analyze why they disagree and synthesize the best answer."
            )
        ])
        
        return FinalAnswer(
            answer=self._extract_answer(disagreement_analysis),
            reasoning=disagreement_analysis,
            confidence=self._calculate_confidence(solutions, disagreement_analysis)
        )
```

---

## 📊 Phase 5: Real Evaluation (No More Fake Tests)

### Benchmark on Real Queries
```python
# tests/test_real_performance.py
class TestRealPerformance:
    """Tests that use REAL LLM and measure ACTUAL performance."""
    
    def test_math_accuracy_on_real_problems(self, real_llm_client):
        """Test MathWorker on actual math problems."""
        math_worker = MathWorker()
        math_worker.llm_client = real_llm_client
        
        # Real math problems with known answers
        problems = [
            ("What is 17 * 23?", "391"),
            ("Solve for x: 2x + 5 = 17", "x = 6"),
            ("What is the derivative of x^3 + 2x?", "3x^2 + 2"),
        ]
        
        correct = 0
        for query, expected in problems:
            result = math_worker.solve(query)
            if self._answers_match(result.answer, expected):
                correct += 1
        
        accuracy = correct / len(problems)
        print(f"Math Worker Accuracy: {accuracy:.1%}")
        
        # Test should pass if accuracy > 80%
        assert accuracy > 0.8, f"Accuracy too low: {accuracy:.1%}"
    
    def test_meta_reasoner_improves_accuracy(self, real_llm_client):
        """Verify meta-reasoner actually improves over single workers."""
        
        # Create workers
        math_worker = MathWorker()
        logic_worker = LogicWorker()
        
        # Test on ambiguous problems
        ambiguous_problems = [
            ("If John has 5 apples and gives 2 to Mary, how many does John have?", "3"),
            # More problems...
        ]
        
        # Test single worker
        single_accuracy = self._test_worker(math_worker, ambiguous_problems)
        
        # Test meta-reasoner with debate
        meta = SmartMetaReasoner(real_llm_client)
        meta.add_workers([math_worker, logic_worker])
        meta_accuracy = self._test_meta(meta, ambiguous_problems)
        
        improvement = meta_accuracy - single_accuracy
        print(f"Single Worker: {single_accuracy:.1%}")
        print(f"Meta-Reasoner: {meta_accuracy:.1%}")
        print(f"Improvement: {improvement:+.1%}")
        
        # Meta-reasoner MUST improve over single worker
        assert improvement > 0.1, f"Meta-reasoner didn't improve: {improvement:+.1%}"
```

---

## 🎯 Success Criteria

### Must Pass (No Compromise)
1. ✅ All tests use real LLM - no mocks anywhere
2. ✅ Tests fail with clear message if LLM offline
3. ✅ Router uses embeddings, not keywords
4. ✅ Workers use different reasoning methods (not just prompts)
5. ✅ Meta-reasoner shows measurable improvement (>10%) over single workers
6. ✅ Workers actually debate and refine through multiple rounds

### Innovation Metrics
- **Router Accuracy**: >85% (measured on held-out test set)
- **Meta-Reasoner Improvement**: >15% over best single worker
- **Debate Convergence**: <3 rounds to reach consensus
- **Real-World Performance**: Tested on actual user queries

---

## 🚀 Implementation Order

1. **NOW**: Remove all mocks, create real LLM fixture
2. **Week 1**: Implement embedding-based router
3. **Week 2**: Enhance worker specialization (formal logic, code execution, RAG)
4. **Week 3**: Implement debate-based meta-reasoner
5. **Week 4**: Real-world evaluation and iteration

---

## 📝 Notes

- Tests will be slower (5-10s vs 0.1s) - **THAT'S GOOD**
- Tests will fail more - **THAT'S FEEDBACK**
- We'll see what actually works - **THAT'S LEARNING**

**This is what building a real product looks like.**
