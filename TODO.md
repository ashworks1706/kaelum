# KaelumAI Development TODO

## Sprint 1: Core MVP (Week 1-2)

### LLM Integration
- [x] Basic OpenAI-compatible client
- [x] Ollama support
- [ ] vLLM support
- [ ] Error handling
- [ ] Retry logic

### Reasoning
- [x] Basic reasoning trace generation
- [ ] Mode templates (math, code, logic)
- [ ] Structured output parsing
- [ ] Chain-of-thought prompting

### Verification
- [x] Symbolic verification (SymPy)
- [ ] Better equation parsing
- [ ] Multi-step verification
- [ ] Verification error messages

### API
- [x] Basic `enhance()` function
- [ ] Configuration system
- [ ] Response formatting
- [ ] Error handling

## Sprint 2: Advanced Features (Week 3-4)

### RAG Verification
- [x] RAG adapter interface
- [x] ChromaDB adapter
- [x] Qdrant adapter
- [ ] Test with real databases
- [ ] Similarity threshold tuning
- [ ] Context management

### Self-Correction
- [x] Basic reflection loop
- [ ] Adaptive stopping (confidence-based)
- [ ] Better reflection prompts
- [ ] Issue tracking

### Confidence Scoring
- [ ] Pattern-based scoring
- [ ] Verification-based scoring
- [ ] Combine multiple signals
- [ ] Calibration

## Sprint 3: Optimization (Week 5-6)

### Performance
- [ ] Caching layer (LRU)
- [ ] Redis support
- [ ] Batch processing
- [ ] Async operations

### Tool Selection
- [ ] Tool registry
- [ ] Selection logic
- [ ] Guardrails
- [ ] Fallback handling

### Agent Orchestration
- [ ] Multi-agent coordination
- [ ] Task decomposition
- [ ] Agent selection
- [ ] Result aggregation

## Sprint 4: Benchmarks (Week 7-8)

### Benchmark Suite
- [ ] Speed tests
- [ ] Hallucination detection tests
- [ ] Tool selection tests
- [ ] Math reasoning tests
- [ ] Agent orchestration tests

### Metrics
- [ ] Latency tracking
- [ ] Cost tracking
- [ ] Accuracy measurement
- [ ] Visualization

### Comparison
- [ ] Baseline (raw LLM)
- [ ] With KaelumAI
- [ ] Different models
- [ ] Different techniques

## Research to Implement

### Priority Papers
- [ ] Chain-of-Verification (CoVe) - Meta
- [ ] Self-Consistency - Google
- [ ] ReAct - Princeton/Google
- [ ] Tree-of-Thoughts (ToT) - Princeton
- [ ] Program-Aided Language Models (PAL) - CMU
- [ ] Verify-and-Edit - OpenAI

### Assignment (Pick 3 each)
- [ ] Ash: 
- [ ] r3tr0:
- [ ] wsb:

## Done ✅

- [x] Basic project structure
- [x] LLM client (OpenAI-compatible)
- [x] Symbolic verification
- [x] RAG adapter interface
- [x] Basic reflection
- [x] Simple orchestration
- [x] One-line API

## Backlog / Future

- [ ] Streaming support
- [ ] Web dashboard
- [ ] LangChain integration
- [ ] LlamaIndex integration
- [ ] CLI tool
- [ ] Docker deployment
- [ ] API server
- [ ] Documentation site
