# Kaelum Frontend Guide - Educational Research Dashboard

## 🎯 Overview

The Kaelum frontend is a comprehensive **educational research platform** that provides:
- **Real-time visualization** of AI reasoning processes
- **Live monitoring** of LATS tree search, neural routing, and verification
- **Interactive configuration** for experimentation
- **Fine-tuning data collection** interface
- **Complete transparency** into system operations for learning and research

## 🚀 Quick Start

### Option 1: Automated Start (Recommended)
```bash
./start_demo.sh
```

### Option 2: Manual Start
```bash
# Terminal 1 - Backend (Flask API)
cd backend
python app.py

# Terminal 2 - Frontend (Next.js)
cd frontend
npm install  # First time only
npm run dev
```

Then open: **http://localhost:3000**

## 📊 Dashboard Components

### 1. 💬 Query Interface
**Purpose:** Interactive testing ground for the reasoning system

**Features:**
- Submit natural language queries
- See real-time processing
- View reasoning steps with detailed explanations
- Metadata display: worker type, confidence, verification status, execution time
- Cache hit indicators
- Example queries for different domains

**Educational Value:**
- Understand how LATS explores multiple reasoning paths
- See which expert worker handles different query types
- Observe verification and reflection in action

---

### 2. 📝 Live Logs
**Purpose:** Real-time system logging for complete transparency

**Features:**
- Streaming logs from all components:
  - Orchestrator (main pipeline)
  - Neural Router (worker selection)
  - LATS (tree search operations)
  - Workers (Math, Code, Logic, etc.)
  - Verification (SymPy, AST, semantic checks)
  - Cache (similarity lookups, LLM validation)
  - Reflection (self-correction)
- Filter by log level: INFO, WARNING, ERROR, DEBUG
- Auto-scroll toggle for continuous monitoring
- Pause/Resume streaming
- Timestamp tracking

**Educational Value:**
- See the complete pipeline in action
- Understand component interactions
- Debug and learn from system behavior
- Track LATS pruning decisions
- Monitor cache validation logic

---

### 3. 📊 Metrics Dashboard
**Purpose:** Comprehensive performance analytics

**Features:**
- **Overall Stats:**
  - Total queries processed
  - Success rate
  - Average execution time
  - Cache hit rate

- **LATS Performance:**
  - Average nodes explored
  - Average iterations (reflection cycles)
  - Search efficiency metrics

- **Per-Worker Analytics:**
  - Queries handled by each expert
  - Success rate per worker
  - Average reward scores
  - Execution time per worker

- **Verification Engine:**
  - Total verifications
  - Pass/fail breakdown
  - Pass rate trends

- **Reflection System:**
  - Self-correction attempts
  - Average iterations per query
  - Improvement rate (% fixed after reflection)

**Educational Value:**
- Understand system performance characteristics
- Compare worker effectiveness
- See verification's impact on quality
- Learn how reflection improves accuracy

---

### 4. 🧠 Neural Router
**Purpose:** Visualize AI-powered query routing

**Features:**
- **Training Status:**
  - Total queries processed
  - Model trained status (trains every 32 queries)
  - Next training countdown
  - Overall routing success rate

- **Worker Distribution:**
  - Bar chart showing query distribution across workers
  - Percentage breakdown
  - Color-coded by worker type

- **Recent Routing Decisions:**
  - Last 5 routing decisions with outcomes
  - Confidence scores
  - Latency tracking

- **Learning Explanation:**
  - How neural network works (398→256→128 architecture)
  - Feature extraction details (embeddings + structural features)
  - Training process with enhanced feedback

**Educational Value:**
- See machine learning in action
- Understand how router improves over time
- Learn about neural network architecture
- Observe continual learning

---

### 5. ⚡ Smart Cache
**Purpose:** Demonstrate semantic caching with LLM validation

**Features:**
- **Cache Statistics:**
  - Total cached LATS trees
  - LLM validations performed
  - Acceptance/rejection rates

- **Validation Breakdown:**
  - ✓ Accepted: Queries reusing cached solutions
  - ✗ Rejected: False positives prevented by LLM
  - Visual progress bar

- **Cached Trees Display:**
  - Preview of cached queries
  - Worker type and tree size
  - Cache IDs for tracking

- **System Explanation:**
  - Two-stage validation process
  - Embedding similarity (fast pre-filter)
  - LLM semantic validation (accurate check)

**Educational Value:**
- Learn about semantic similarity
- Understand false positive prevention
- See speedup vs accuracy tradeoff
- Observe LLM-powered validation

---

### 6. ⚙️ Configuration Panel
**Purpose:** Interactive system configuration for experimentation

**Features:**
- **LLM Backend:**
  - Base URL (vLLM/Ollama/OpenAI endpoint)
  - Model name selection
  - Temperature control (0.0-2.0)
  - Max tokens setting

- **Neural Router & Embeddings:**
  - Embedding model selection (different dimensions/speed)
  - Enable/disable router
  - Router behavior control

- **Verification Settings:**
  - Toggle symbolic verification (SymPy math)
  - Toggle factual verification (semantic checks)
  - Max reflection iterations (self-correction cycles)

- **LATS Search:**
  - Enable/disable parallel search
  - Max workers for parallelization
  - Performance tuning

**Educational Value:**
- Experiment with different models
- Understand parameter impact on quality
- Learn about verification tradeoffs
- Test performance optimizations

---

### 7. 🎯 Fine-tuning Panel
**Purpose:** Collect and export data for model fine-tuning

**Features:**
- **Selection Strategies:**
  1. **Uncertainty Sampling**: Queries with low confidence
  2. **Diversity Sampling**: Semantically diverse queries
  3. **Error-Based Sampling**: Failed verifications
  4. **Complexity Sampling**: High reasoning complexity
  5. **Mixed Strategy**: Balanced combination (recommended)

- **Export Configuration:**
  - Batch size selection (1-100)
  - Custom export path
  - One-click data export

- **Fine-tuning Workflow:**
  - Step-by-step guide
  - Command examples for training
  - Script documentation
  - Tips and best practices

**Educational Value:**
- Learn about active learning
- Understand fine-tuning workflows
- See how to specialize models
- Practice continual improvement

---

### 8. 🏗️ Architecture
**Purpose:** Visual explanation of system design

**Features:**
- **Processing Pipeline:**
  - 6-step flow diagram with descriptions
  - Color-coded stages
  - Clear progression arrows

- **Core Technologies:**
  - Neural Router explanation
  - LATS (Monte Carlo Tree Search)
  - Multi-layer verification
  - Semantic cache + LLM validation
  - Reflection engine
  - Active learning

- **Research Contributions:**
  - Enhanced router feedback
  - Early LATS pruning
  - Two-stage cache validation
  - Automated reflection loop

**Educational Value:**
- Understand system architecture
- Learn how components interact
- See research innovations
- Grasp theoretical foundations

---

## 🔧 Technical Architecture

### Backend (Flask - Port 5000)
```
API Endpoints:
├── /api/health          - Health check
├── /api/config          - GET/POST configuration
├── /api/query           - Process reasoning queries
├── /api/metrics         - System-wide metrics
├── /api/stats/router    - Router analytics
├── /api/stats/cache     - Cache statistics
├── /api/stats/calibration - Threshold stats
├── /api/export/training-data - Export fine-tuning data
└── /api/workers         - List available workers
```

### Frontend (Next.js - Port 3000)
```
Components:
├── page.tsx                 - Main dashboard with tabs
├── components/
│   ├── QueryInterface.tsx   - Interactive query UI
│   ├── LogViewer.tsx        - Live log streaming
│   ├── MetricsDashboard.tsx - Performance analytics
│   ├── RouterVisualization.tsx - Router stats
│   ├── CacheVisualization.tsx - Cache analytics
│   ├── ConfigPanel.tsx      - Settings control
│   ├── FineTuningPanel.tsx  - Data export UI
│   └── SystemArchitecture.tsx - Architecture diagram
```

## 📚 Learning Paths

### For Beginners
1. Start with **Architecture** tab to understand the system
2. Try **Query Interface** with example queries
3. Watch **Live Logs** to see processing in real-time
4. Check **Metrics** to understand performance

### For Intermediate Users
1. Experiment with **Configuration** settings
2. Observe **Neural Router** learning over time
3. Study **Cache** validation decisions
4. Analyze worker-specific performance in **Metrics**

### For Advanced Users
1. Collect diverse queries for fine-tuning
2. Export data with different **strategies**
3. Fine-tune specialized models
4. Benchmark custom configurations

## 🎨 Design Philosophy

### Transparency First
- Every decision is visible (routing, caching, verification)
- Logs show complete reasoning process
- Metrics expose all performance characteristics

### Educational Focus
- Clear explanations throughout UI
- "How It Works" sections in each component
- Visual diagrams and color coding
- Progressive disclosure of complexity

### Research Enablement
- All data exportable for analysis
- Configuration flexibility for experimentation
- Real-time monitoring for debugging
- Historical tracking of improvements

## 🐛 Troubleshooting

### Backend Not Starting
```bash
# Check if port 5000 is in use
lsof -i :5000

# Run with debug logging
cd backend
python app.py
```

### Frontend Build Errors
```bash
cd frontend
rm -rf node_modules .next
npm install
npm run dev
```

### API Connection Failed
1. Ensure backend is running: `curl http://localhost:5000/api/health`
2. Check CORS is enabled in `backend/app.py`
3. Verify no firewall blocking localhost

### No Metrics Displayed
1. Run at least one query through the system
2. Wait for data collection (metrics update every 5 seconds)
3. Check `.kaelum/` directory exists and has data

## 📖 Related Documentation

- **Main README.md**: Complete system documentation
- **finetune_setup.py**: Fine-tuning script usage
- **export_cache_validation_data.py**: Cache validation export
- **core/**: Source code with inline documentation

## 🎓 Educational Use Cases

### Computer Science Classes
- **AI/ML**: See neural networks, MCTS, active learning
- **Software Engineering**: Observe modular architecture, API design
- **Algorithms**: Study tree search, pruning, optimization

### Research Projects
- **Reasoning Systems**: Benchmark different approaches
- **Model Evaluation**: Compare worker performance
- **Cache Strategies**: Test validation techniques

### Self-Learning
- **Interactive Exploration**: Hands-on experimentation
- **Visual Learning**: See abstract concepts in action
- **Iterative Improvement**: Track learning over time

## 🚀 Future Enhancements

Potential additions for educational value:
- [ ] WebSocket for real-time log streaming
- [ ] Interactive LATS tree visualization
- [ ] A/B testing framework for configurations
- [ ] Jupyter notebook integration
- [ ] Export reports as PDF/HTML
- [ ] Historical comparison charts
- [ ] Annotation/note-taking in UI
- [ ] Shared experiment sessions

## 💡 Tips for Best Experience

1. **Start Simple**: Use small models (1.7B-3B) for fast iteration
2. **Enable All Logs**: Set DEBUG level to see everything
3. **Run Diverse Queries**: Test all 6 worker types
4. **Monitor Training**: Watch router improve after 32 queries
5. **Export Regularly**: Collect fine-tuning data incrementally
6. **Document Experiments**: Take notes on configuration impacts

---

**Built for transparency, learning, and research** 🎓✨
