# Hardcoded Values and Duct-Tape Issues

This document lists all the hardcoded values, magic numbers, and temporary workarounds found in the codebase. These should be made configurable or properly documented.

## Critical Issues

### 1. Bare Exception Handlers (Silent Failures)
These catch all exceptions and either pass or do nothing, which can hide bugs:

**Location**: Throughout the codebase (20+ instances)
- `core/workers/workers.py:53` - LATS tree serialization failure silently ignored
- `core/workers/workers.py:217, 261, 397` - Multiple bare except blocks
- `core/workers/analysis_worker.py:92`
- `core/workers/creative_worker.py:100`
- `core/workers/code_worker.py:111`
- `core/workers/factual_worker.py:111`
- `core/reasoning.py:33` - LLM call failure silently ignored
- `core/detectors/conclusion_detector.py:14, 18, 137, 156` - 4 bare excepts
- `core/detectors/coherence_detector.py:15, 79`
- `core/detectors/repetition_detector.py:174`
- `core/detectors/completeness_detector.py:14, 21, 135, 177`
- `backend/app.py:36, 257, 278, 317, 326, 349` - 6 bare excepts

**Issue**: These should at least log the error or catch specific exceptions.

**Recommendation**: 
```python
# Bad
except:
    pass

# Good
except Exception as e:
    logger.warning(f"Failed to serialize tree: {e}")
```

---

## Hardcoded URLs and Ports

### 2. Backend URL Hardcoding
**Files**: Frontend components and backend config

**Issues**:
- `frontend/components/QueryInterface.tsx:46-47` - Hardcoded `http://localhost:5000`
- `frontend/components/FeedbackPanel.tsx:92-93` - Hardcoded `http://localhost:5000`
- `frontend/components/FineTuningPanel.tsx:55` - Hardcoded `http://localhost:5000`
- `frontend/components/MetricsDashboard.tsx:59` - Hardcoded `http://localhost:5000`
- `frontend/components/RouterVisualization.tsx:35` - Hardcoded `http://localhost:5000`
- `frontend/components/CacheVisualization.tsx:27` - Hardcoded `http://localhost:5000`
- `frontend/components/ConfigPanel.tsx:35, 53` - Hardcoded `http://localhost:5000`
- `frontend/app/page.tsx:21, 27` - Hardcoded `http://localhost:5000`
- `backend/app.py:673` - Hardcoded port 5000: `app.run(..., port=5000)`
- `backend/config.py:11` - `"base_url": "http://localhost:8000/v1"`
- `core/config.py:6` - `base_url: str = Field(default="http://localhost:8000/v1")`

**Recommendation**: Use environment variables or a config file for all URLs/ports.

---

## Magic Numbers in Algorithms

### 3. LATS Pruning Thresholds
**File**: `core/search/lats.py:171`

```python
if cur.visits >= 3 and avg_reward < 0.3:
```

**Issue**: Values `3` (min visits) and `0.3` (reward threshold) are hardcoded.

**Recommendation**: Make these configurable parameters with defaults:
```python
@dataclass
class LATSConfig:
    prune_min_visits: int = 3
    prune_reward_threshold: float = 0.3
```

---

### 4. Cache Similarity Thresholds
**File**: `core/search/tree_cache.py:22-29`

```python
WORKER_THRESHOLDS = {
    "math": 0.90,
    "code": 0.87,
    "logic": 0.88,
    "factual": 0.80,
    "creative": 0.75,
    "analysis": 0.82
}
```

**Issue**: These per-worker thresholds are hardcoded.

**Also**: `tree_cache.py:62` - Default threshold `0.85`
**Also**: `tree_cache.py:326, 331` - Validation thresholds `0.95` and `0.90`

**Recommendation**: Move to configuration with explanation of why each worker has different thresholds.

---

### 5. Router Neural Network Architecture
**File**: `core/search/router.py:109`

```python
def __init__(self, input_dim: int = 398, hidden_dim: int = 256):
```

**Issue**: Architecture dimensions `398`, `256`, `128` are hardcoded.

**File**: `core/search/router.py:247, 270-272`
```python
adjusted_probs = worker_probs + feedback_tensor * 0.3
max_tree_depth = int(torch.clamp(outputs['depth_logits'], 3, 10).item())
num_simulations = int(torch.clamp(outputs['sims_logits'], 5, 25).item())
use_cache = torch.sigmoid(outputs['cache_logits']).item() > 0.5
```

**Issue**: 
- Feedback multiplier `0.3`
- Depth range `[3, 10]`
- Simulation range `[5, 25]`
- Cache threshold `0.5`

**Recommendation**: Make architecture configurable or at least document why these specific values.

---

### 6. Router Training Parameters
**File**: `core/search/router.py:157`

```python
buffer_size: int = 32
learning_rate: float = 0.001
exploration_rate: float = 0.1
```

**File**: `core/search/router.py:364, 370`
```python
high_quality_samples = [item for item in self.training_buffer if item.get("quality", 0.0) > 0.8]
```

**Issue**: Quality threshold `0.8` for training sample selection is hardcoded.

**File**: `core/search/router.py:396`
```python
total_loss = worker_loss + 0.1 * depth_loss + 0.1 * sims_loss + 0.05 * cache_loss
```

**Issue**: Loss weights `0.1`, `0.1`, `0.05` are hardcoded.

---

### 7. Reward Model Configurations
**File**: `core/search/reward_model.py:19-57`

All reward weights are hardcoded in a dictionary:

```python
CONFIGS = {
    "math": {
        "has_answer": 0.85,
        "partial": 0.50,
        "base": 0.40,
        "depth_penalty": 0.06
    },
    # ... 5 more workers
}
```

**Issue**: These are tuned values but not documented why these specific numbers.

**File**: `core/search/reward_model.py:62, 67`
```python
return 0.3  # Default confidence
confidence = (exploration_confidence * 0.3 + answer_quality * 0.7)
```

**Issue**: Confidence calculation weights `0.3` and `0.7` are hardcoded.

---

### 8. Feature Extraction Normalization
**File**: `core/search/router.py:90-99`

```python
self.word_count / 50.0,
self.question_words / 5.0,
self.avg_word_length / 10.0,
self.punctuation_count / 10.0,
```

**Issue**: Normalization factors `50.0`, `5.0`, `10.0`, `10.0` are hardcoded. No explanation of why these values.

---

## Configuration Issues

### 9. Default Configuration Values
**File**: `backend/config.py:10-29`

```python
DEFAULT_CONFIG = {
    "base_url": "http://localhost:8000/v1",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "api_key": "EMPTY",  # ← Suspicious
    "temperature": 0.7,
    "max_tokens": 512,
    # ...
}
```

**Issue**: `"api_key": "EMPTY"` is a string instead of None. This gets sent to the API which might cause issues.

**Recommendation**: Use `None` or `""` instead of `"EMPTY"`.

---

### 10. Torch CUDA Hack
**File**: `backend/app.py:30-36`

```python
try:
    import torch
    torch.set_default_device('cpu')
    if torch.cuda.is_available():
        torch.cuda.is_available = lambda: False
    print(">>> torch configured", flush=True)
except ImportError:
    print(">>> torch not available", flush=True)
    pass
```

**Issue**: Overriding `torch.cuda.is_available` is a hack to force CPU usage. This is fragile and could break.

**Recommendation**: Use `CUDA_VISIBLE_DEVICES=""` environment variable or `torch.device('cpu')` properly.

---

## Debug Code Left In Production

### 11. Print Statements Instead of Logging
**Files**: Multiple

- `core/cache_validator.py:140, 152, 179, 181` - Print statements in production code
- `core/verification/sympy_engine.py:17` - Debug prints
- `core/verification/verification.py:24, 154` - Debug prints
- `backend/app.py:33, 38-40` - Startup prints (these are okay but inconsistent)

**Recommendation**: Replace all `print()` with proper logging:
```python
# Bad
print(f"Warning: Failed to log validation: {e}")

# Good
logger.warning(f"Failed to log validation: {e}")
```

---

## Missing Error Handling

### 12. Functions Returning None on Error
Multiple functions return `None` on error without logging:

- `core/cache_validator.py:153`
- `core/workers/code_worker.py:198, 275`
- `core/workers/workers.py:129, 138, 144`
- `core/detectors/completeness_detector.py:122`
- `core/search/tree_cache.py:248, 273, 317, 353`
- `core/search/lats.py:184`
- `core/verification/threshold_calibrator.py:92`
- `core/verification/reflection.py:85, 98, 103`

**Issue**: Calling code can't distinguish between "no result found" and "error occurred".

**Recommendation**: Either raise exceptions or return `(result, error)` tuples.

---

## Summary of Recommended Actions

### High Priority
1. **Fix all bare exception handlers** - Add proper logging at minimum
2. **Make URLs/ports configurable** - Use environment variables
3. **Fix torch CUDA hack** - Use proper device management
4. **Remove "EMPTY" API key** - Use None or empty string

### Medium Priority
5. **Document all magic numbers** - Add comments explaining thresholds
6. **Make algorithm parameters configurable** - Create config dataclasses
7. **Replace print with logging** - Consistent logging throughout

### Low Priority
8. **Improve error handling** - Return proper error types
9. **Document reward configurations** - Explain why these specific values
10. **Add configuration validation** - Ensure values are in valid ranges

---

## Notes

Most of these hardcoded values are from tuning/experimentation and probably work well. The issue is they're not documented or configurable, making it hard to:
- Understand why these values were chosen
- Experiment with different values
- Deploy in different environments
- Debug when things go wrong (silent failures)

The bare exception handlers are the biggest concern - they hide real bugs.

---

# Overengineered / Unused Components

This section lists components that are either unnecessarily complex for the current use case, not actually used, or could be significantly simplified.

## Completely Unused Components

### 1. RepetitionDetector (321 lines)
**File**: `core/detectors/repetition_detector.py`

**What it does**: 
- Loads SentenceTransformer model
- Uses TfidfVectorizer
- Has 3 different types of repetition detection (semantic, lexical, phrase)
- Includes "stylistic pattern detection" with ML
- Has adaptive thresholds and calibration

**Problem**: **NOT USED ANYWHERE IN THE CODEBASE**

Search results show it's never imported or instantiated in any worker, orchestrator, or runtime code.

**Recommendation**: Delete this file entirely. If repetition detection is needed later, start with a simple 20-line version.

---

### 2. IsotonicRegression Import
**File**: `core/detectors/task_classifier.py:4`

```python
from sklearn.isotonic import IsotonicRegression
```

**Problem**: Imported but never used. Not instantiated anywhere in the file.

**Recommendation**: Remove the import.

---

## Overengineered Components

### 3. Task Classifier with 100+ Exemplars (314 lines)
**File**: `core/detectors/task_classifier.py`

**What it does**:
- Has 6 task types for code alone (debugging, optimization, review, testing, algorithm, generation)
- Each task type has 6 hardcoded exemplar strings
- Loads SentenceTransformer
- Has adaptive thresholds, calibration, performance history
- Uses sklearn IsotonicRegression (imported but not used)

**Current Usage**: Only used in `code_worker.py` and `factual_worker.py`

**Problem**: 
- The 100+ hardcoded exemplar strings are basically just keyword matching with extra steps
- SentenceTransformer embedding of 6 sentences could be replaced with simple regex
- The complexity doesn't add value for current use case

**Recommendation**: Replace with simple keyword/regex matching:
```python
class SimpleTaskClassifier:
    PATTERNS = {
        'debugging': r'\b(bug|fix|error|debug|broken|issue)\b',
        'testing': r'\b(test|pytest|unittest|assert)\b',
        # ... 10 lines total
    }
```

---

### 4. Multiple Calibrators (400+ lines combined)
**Files**: 
- `core/verification/threshold_calibrator.py` (152 lines)
- `core/verification/confidence_calibrator.py` (131 lines)
- Plus calibration logic in detectors

**What they do**:
- Track decisions over time
- Compute optimal thresholds using F1 score grid search
- Calibrate confidence scores with binning
- Apply feature-based adjustments

**Problem**:
- Requires 20+ samples before doing anything
- The "optimal" threshold is just trying 0.2, 0.25, 0.3, ... 0.85
- Most of the complexity is for handling edge cases that rarely happen
- The confidence calibrator bins predictions into 10 buckets but you need 200+ samples for this to be meaningful

**Recommendation**: 
- Start with fixed thresholds (0.5 is fine)
- Add simple moving average if really needed
- Current implementation is premature optimization

---

### 5. Human Feedback System (382 lines)
**File**: `core/learning/human_feedback.py`

**What it does**:
- Tracks worker accuracy, corrections, preferences
- Adjusts worker probabilities
- Adjusts reward multipliers for reasoning steps
- Stores detailed statistics
- Has comprehensive dataclasses for feedback

**Problem**:
- The README presents this as a key feature
- But in practice, how many users will actually provide detailed feedback?
- 382 lines for a feature that requires manual user input after every query
- Most users just want an answer, not to train the system

**Reality Check**:
- Academic/research setting: Maybe 5-10% of queries get feedback
- Production setting: <1% of queries get feedback
- The system doesn't degrade without it, so it's optional

**Recommendation**: This is fine to keep if it's genuinely a learning project about feedback systems. But be realistic about calling it "continuous improvement" when it requires manual input.

---

### 6. Conclusion Detector with BART Model (320 lines)
**File**: `core/detectors/conclusion_detector.py`

**What it does**:
- Tries to load `facebook/bart-large-mnli` (1.4GB model!)
- Falls back to lighter models
- Has hardcoded conclusion/non-conclusion exemplars
- Uses zero-shot classification
- Has rule-based patterns, embedding similarity, and ML classification

**Problem**:
- Downloads a 1.4GB model just to detect if text has a conclusion
- Has 3 different detection methods (rules, embeddings, ML) when rules alone would work
- The exemplars are: "Therefore, we can conclude...", "In conclusion...", "Hence..."
- Could be replaced with: `if re.search(r'\b(therefore|thus|hence|in conclusion)\b', text, re.I)`

**Used in**: `analysis_worker.py`, `factual_worker.py`, `workers.py` base class

**Recommendation**: Replace with simple pattern matching. If ML is really needed, use the already-loaded sentence transformer, not a separate 1.4GB model.

---

### 7. Coherence Detector (148 lines)
**File**: `core/detectors/coherence_detector.py`

**What it does**:
- Loads another `facebook/bart-large-mnli` model instance
- Computes sentence-level coherence with embeddings
- Has NLI (Natural Language Inference) pipeline

**Used in**: `creative_worker.py` only

**Problem**:
- Creative worker is for creative writing/brainstorming
- "Coherence" for creative output is subjective
- Loading a 1.4GB model for this is overkill
- Simpler metrics (sentence length variance, vocabulary diversity) would work

**Recommendation**: Use simple heuristics or remove entirely.

---

### 8. Completeness Detector (276 lines)
**File**: `core/detectors/completeness_detector.py`

**What it does**:
- Yet another `facebook/bart-large-mnli` model
- Checks if answer is "complete" using NLI
- Has question classification, information coverage scoring
- Attempts to validate "factual coverage"

**Used in**: `factual_worker.py` only

**Problem**:
- "Completeness" is subjective and context-dependent
- The NLI model can't actually verify facts
- This is pretending to do semantic fact-checking without a knowledge base

**Recommendation**: Either integrate a real knowledge base or remove this. Current implementation gives false sense of validation.

---

### 9. Cache Validator with LLM (181 lines + training data export)
**File**: `core/cache_validator.py`

**What it does**:
- For cache hits, asks the LLM: "Would cached answer satisfy new query?"
- Logs all validation decisions to JSONL
- Has export function for "training data"
- Plans to fine-tune a validator model

**Problem**:
- You're calling the LLM to validate cache hits, which adds latency (0.1-0.3s)
- The point of a cache is to be fast
- "Training a validator" requires 100s of examples and fine-tuning infrastructure
- The validation itself requires an LLM call, so you're adding overhead to "save" an LLM call

**Reality Check**:
- Cache hit: 0.001s (fast!)
- Cache hit + LLM validation: 0.1-0.3s (not that fast)
- Just doing the actual query: 2-5s (slow)

**Recommendation**: 
- For 0.85+ similarity, just return the cached result
- For 0.75-0.85 similarity, do a simple keyword check
- Remove the LLM validation entirely

---

## Questionable Design Patterns

### 10. Shared Encoder Pattern (30 lines)
**File**: `core/shared_encoder.py`

**What it does**: Singleton pattern for SentenceTransformer to avoid loading multiple times

**Problem**:
- Despite this pattern, EVERY detector/classifier loads its own SentenceTransformer instance
- `RepetitionDetector.__init__`: `self.encoder = SentenceTransformer(embedding_model)`
- `ConclusionDetector.__init__`: `self.encoder = SentenceTransformer(embedding_model)`
- `TaskClassifier.__init__`: `self.encoder = SentenceTransformer(embedding_model)`
- None of them use `get_shared_encoder()`

**Recommendation**: Either use the shared encoder everywhere or remove the pattern.

---

### 11. ThresholdCalibrator Instances Everywhere
**Files**: Imported in 7+ detector files

Each detector creates its own `ThresholdCalibrator()` instance:
- `RepetitionDetector`: `self.threshold_calibrator = ThresholdCalibrator()`
- `ConclusionDetector`: `self.threshold_calibrator = ThresholdCalibrator()`
- `TaskClassifier`: `self.threshold_calibrator = ThresholdCalibrator()`
- And so on...

**Problem**:
- Each calibrator needs 20+ samples to compute optimal thresholds
- If you have 7 calibrator instances, you need 140+ samples
- They don't share data, so each learns independently
- This defeats the purpose of calibration

**Recommendation**: One global calibrator or none at all.

---

## Summary Stats

**Total lines in detectors/**: ~2,200 lines
**Actually necessary**: ~200 lines (10%)

**Large ML models loaded**:
- `facebook/bart-large-mnli`: 1.4GB (loaded 3-5 times!)
- `all-MiniLM-L6-v2`: 80MB (loaded 7+ times)

**Complexity by LOC**:
1. verification.py: 592 lines (mostly SymPy, this is justified)
2. human_feedback.py: 382 lines (optional feature)
3. repetition_detector.py: 321 lines (UNUSED!)
4. conclusion_detector.py: 320 lines (could be 20 lines)
5. task_classifier.py: 314 lines (could be 30 lines)

---

## Recommendations by Priority

### Immediate (Delete Dead Code)
1. **Delete RepetitionDetector** - 321 lines of unused code
2. **Remove IsotonicRegression import** - imported but never used

### High Priority (Simplify Overengineered)
3. **Replace conclusion detection** - Pattern matching instead of 1.4GB model
4. **Simplify task classification** - Regex instead of 100+ exemplars
5. **Remove or simplify cache validator** - LLM validation defeats the cache purpose

### Medium Priority (Consider Removing)
6. **Coherence detector** - Subjective and adds little value
7. **Completeness detector** - Can't actually verify facts without knowledge base
8. **Multiple calibrators** - Need 100s of samples, premature optimization

### Low Priority (Architectural)
9. **Fix shared encoder pattern** - Either use it or remove it
10. **Consider feedback system scope** - 382 lines for manual feature

---

## The Core Issue

The system has **two** philosophies mixed together:

**Philosophy A**: "Let's build a research platform with all the bells and whistles"
- Multiple ML models for detection/classification
- Calibration systems that learn over time
- Human feedback loops
- Active learning and fine-tuning pipelines

**Philosophy B**: "Let's build a working AI reasoning system"
- LATS tree search (this works!)
- Domain-specific workers (this works!)
- SymPy verification (this works!)
- Neural router (this works!)

**The Problem**: Philosophy A components add ~3,000 lines of code and multiple heavy ML models, but don't significantly improve the core reasoning quality. They're research scaffolding presented as production features.

---

## What Actually Matters

The core value is:
1. **LATS search** - Explores multiple reasoning paths
2. **Worker specialization** - Math, Code, Logic, etc.
3. **Symbolic verification** - Catches math errors
4. **Neural router** - Learns which worker to use

Everything else is nice-to-have at best, unused at worst.

---

## Realistic Refactor

If this were a production system, you could:
- **Remove**: 2,000+ lines of detector/calibrator code
- **Keep**: 1,500 lines of core LATS/workers/verification
- **Simplify**: Replace ML detectors with 200 lines of regex/heuristics

**Result**: 
- Same reasoning quality
- 10x faster initialization (no heavy model loading)
- 10x easier to understand and maintain
- Still a great learning project for MCTS and reasoning systems

But if the goal is to learn about all these different ML techniques, then the current complexity is justified as educational exploration.

---

## Update: Cleanup Completed (February 2026)

Some of the critical issues documented above have been addressed:

### Fixed ✓
1. **Bare Exception Handlers** - Fixed 15+ instances with proper logging
   - `core/workers/workers.py` (4 instances)
   - `core/workers/code_worker.py` (1 instance) 
   - `core/reasoning.py` (1 instance)
   - `core/detectors/conclusion_detector.py` (3 instances)
   - `core/detectors/completeness_detector.py` (2 instances)
   - `backend/app.py` (4 instances)

2. **Torch CUDA Hack** - Replaced with proper `CUDA_VISIBLE_DEVICES` environment variable

3. **API Key "EMPTY"** - Changed to `None` in `backend/config.py`

4. **Hardcoded Ports** - Backend now uses `BACKEND_HOST` and `BACKEND_PORT` environment variables

5. **Dead Code** - Removed `RepetitionDetector` (321 unused lines) and unused imports

### Still Needs Work
- Magic numbers in LATS, router, and reward models (need documentation)
- Frontend components still have hardcoded `localhost:5000` URLs
- Overengineered detectors with large ML models
- Multiple calibrator instances that need 20+ samples each

See [CLEANUP_COMPLETE.md](CLEANUP_COMPLETE.md) for full details of what was done.
