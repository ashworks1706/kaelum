# Kaelum Frontend

I built this dashboard to visualize what's happening inside the reasoning system. When I was first testing things, I had to read through log files to understand whether the router picked the right worker or if MCTS was pruning effectively. Having a real-time interface made debugging way easier and helped me understand the system behavior better.

The frontend shows live logs from all components, tracks metrics like cache hit rate and router accuracy, and lets you configure everything without editing code. There's also a fine-tuning panel for exporting training data using different active learning strategies.

## Getting Started

**Configuration (optional):**
Backend URL can be configured with environment variable:
```bash
# In frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:5000
```

The frontend automatically detects localhost vs production and adjusts the API URL accordingly.

**Automatic (easiest way):**
```bash
./start_demo.sh
```

**Manual:**
```bash
# Terminal 1 - Backend (default: http://localhost:5000)
cd backend
python app.py

# Or with custom port:
BACKEND_PORT=8080 python app.py

# Terminal 2 - Frontend (default: http://localhost:3000)
cd frontend
npm install  # first time only
npm run dev
```

Then open http://localhost:3000

## Dashboard Tabs

### Query Interface

This is where you test the system. You type a question, submit it, and see the reasoning steps in real-time. It shows which worker handled the query, confidence scores, verification status, execution time, and whether it was a cache hit.

I added example queries for different domains so you can quickly test all six workers without thinking of questions. Watching the reasoning steps unfold helps you understand how LATS explores different solution paths before committing to an answer.

### Live Logs

All components stream their logs here - orchestrator, router, LATS, workers, verification, cache, reflection. You can filter by log level (INFO/WARNING/ERROR/DEBUG) and toggle auto-scroll.

This was super helpful when I was debugging why certain queries failed verification or why the router kept picking the wrong worker. You can see the complete pipeline in action and track individual MCTS pruning decisions.

### Metrics Dashboard

Tracks performance across the whole system:
- Overall: Total queries, success rate, average time, cache hit rate
- LATS: Average nodes explored, iterations per query, search efficiency
- Per-worker: Queries handled, success rate, average rewards, execution time
- Verification: Pass/fail breakdown and trends
- Reflection: Self-correction attempts and improvement rate

The per-worker breakdown is particularly useful for seeing which workers are performing well and which need tuning.

### Neural Router

Shows training status, worker distribution, and recent routing decisions. The router trains every 32 queries, so you can watch the countdown and see when it updates.

There's a bar chart showing query distribution across workers and a list of the last 5 routing decisions with confidence scores. The explanation section describes how the 398→256→128 neural network architecture works and what features get extracted from queries.

### Smart Cache

Displays cache statistics and validation decisions. Shows total cached trees, LLM validations performed, and acceptance vs rejection rates.

The breakdown is interesting because you can see how many false positives the LLM validator prevented. Early versions without LLM validation would serve incorrect answers when embedding similarity was high but queries needed different solutions. The cached trees preview shows what's currently stored.

### Configuration Panel

Lets you change everything without editing config files:
- LLM backend: Base URL, model name, temperature, max tokens
- Neural router & embeddings: Embedding model selection, enable/disable routing
- Verification: Toggle symbolic math verification, factual verification, set max reflection iterations
- LATS search: Enable/disable parallel search, set max workers

I use this constantly for testing different models and parameter combinations. Temperature around 0.7 seems to work well for reasoning tasks.

### Fine-tuning Panel

Exports training data using active learning strategies:
1. **Uncertainty sampling**: Low confidence queries
2. **Diversity sampling**: Semantically diverse queries
3. **Error-based sampling**: Failed verifications
4. **Complexity sampling**: High reasoning complexity
5. **Mixed** (recommended): Balanced combination

You set a batch size (1-100) and export path, then click export. The panel includes a step-by-step fine-tuning workflow with command examples.

I typically use mixed strategy to get a good variety of training examples without manually filtering.

### Architecture

Visual explanation of the processing pipeline (6 stages) and core technologies (MCTS, neural router, verification, cache, reflection, active learning). Shows how components interact and what makes this system different from standard LLM inference.

Good starting point if you want to understand the overall design before diving into specific components.

## Technical Details

**Backend (Flask on port 5000):**
```
/api/health          - Health check
/api/config          - Get/update configuration
/api/query           - Process queries
/api/metrics         - System metrics
/api/stats/router    - Router analytics
/api/stats/cache     - Cache statistics
/api/stats/calibration - Threshold stats
/api/export/training-data - Export fine-tuning data
/api/workers         - List workers
```

**Frontend (Next.js on port 3000):**
```
Components:
├── page.tsx                 - Main dashboard
├── components/
│   ├── QueryInterface.tsx
│   ├── LogViewer.tsx
│   ├── MetricsDashboard.tsx
│   ├── RouterVisualization.tsx
│   ├── CacheVisualization.tsx
│   ├── ConfigPanel.tsx
│   ├── FineTuningPanel.tsx
│   └── SystemArchitecture.tsx
```

## Troubleshooting

**Backend won't start:**
```bash
# Check if port 5000 is in use
lsof -i :5000

# Run with debug logging
cd backend && python app.py
```

**Frontend build errors:**
```bash
cd frontend
rm -rf node_modules .next
npm install
npm run dev
```

**API connection failed:**
- Make sure backend is running: `curl http://localhost:5000/api/health`
- Check CORS is enabled in `backend/app.py`
- Verify no firewall blocking localhost

**No metrics showing:**
- Run at least one query first
- Wait for data collection (updates every 5 seconds)
- Check `.kaelum/` directory exists

---

This dashboard was helpful for understanding what the system was doing under the hood. Reading log files gets tedious quickly, and having live visualization made debugging and experimentation much faster.
