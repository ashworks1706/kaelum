# 🧪 KaelumAI Testing Suite

**All-in-one testing notebook for development and experimentation**

## � The Notebook

### `testing.ipynb`
**Comprehensive testing suite with all experiments in one place**

Organized into 8 sections:
1. **Setup & Configuration** - Switch models, adjust settings
2. **LLM Selection** - Compare speed/quality across models
3. **Benchmark Testing** - GSM8K, TruthfulQA, speed tests
4. **Verification Testing** - SymPy symbolic verification
5. **Reflection Testing** - Self-improvement quality tests
6. **Performance Optimization** - Latency breakdown, token analysis
7. **Integration & Edge Cases** - Real-world scenarios, error handling
8. **Experiment Log** - Document findings, track progress

---

## 🚀 Quick Start

```bash
# Install Jupyter
pip install jupyter

# Launch notebook
cd test_notebooks
jupyter notebook testing.ipynb
```

## 📊 Testing Workflow

**Daily iteration cycle:**
1. Configure model/settings in Section 1
2. Run relevant sections for your task
3. Document findings in Section 8
4. Share results in Discord

**Sequential testing (first run):**
1. Section 2: Choose best LLM
2. Section 3: Establish baseline benchmarks
3. Section 4-5: Test verification + reflection
4. Section 6: Profile performance
5. Section 7: Validate real-world scenarios

## 💡 Tips

- **Start simple**: Use `llama3.2:3b` for fast iteration
- **One variable at a time**: Isolate what you're testing
- **Document in notebook**: Use markdown cells for findings
- **Sequential results**: Easy to compare by scrolling
- **Fast switching**: No need to open multiple files

## 🎯 Success Criteria

From `TODO.md` Sprint 1 targets:
- Speed: < 500ms overhead
- Math: > 90% accuracy (GSM8K)
- Hallucination: > 90% reduction (TruthfulQA)
