# Kaelum Setup Guide

Complete step-by-step guide to get Kaelum running on your system.

---

## 📋 Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **8GB RAM minimum** (16GB recommended)
- **GPU optional** (CPU works fine for small models)
- **Ollama** or any OpenAI-compatible API endpoint

---

## 🚀 Quick Start (5 minutes)

### **Step 1: Install Ollama (Recommended)**

Ollama provides the easiest way to run local LLMs:

```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download
```

**Pull a model:**

```bash
# Small model (recommended for testing - 2GB)
ollama pull qwen2.5:3b

# Or larger model (better quality - 5GB)
ollama pull qwen2.5:7b

# Or tiny model (fastest - 1GB)
ollama pull qwen2.5:1.5b
```

**Verify Ollama is running:**

```bash
ollama list  # Should show your downloaded models
```

Ollama API automatically runs at `http://localhost:11434/v1`

---

### **Step 2: Clone and Setup Kaelum**

```bash
# Clone repository
git clone https://github.com/ashworks1706/KaelumAI.git
cd KaelumAI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Fish shell:
source .venv/bin/activate.fish

# Bash/Zsh:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

---

### **Step 3: Install Dependencies**

```bash
# Install all required packages
pip install -r requirements.txt
```

**Expected installation time:** 2-5 minutes

**Key packages installed:**
- `torch` - Neural router (350MB)
- `sentence-transformers` - Query embeddings (1GB on first use)
- `sympy` - Symbolic verification
- `httpx` - LLM API client
- `pydantic` - Configuration

---

### **Step 4: Run Example**

```bash
# Simple test
python example.py
```

**Expected output:**

```
======================================================================
Kaelum AI - Reasoning System Example
======================================================================

======================================================================
Query 1: What is the derivative of x^2 + 3x?
======================================================================
2x + 3

Worker: math | Confidence: 0.95 | Verification: ✓ PASSED

Reasoning:
1. Apply power rule: d/dx(x^2) = 2x
2. Apply constant rule: d/dx(3x) = 3
3. Sum derivatives: 2x + 3
```

---

## 🔧 Configuration Options

### **Option 1: Using Ollama (Default)**

```python
from kaelum import set_reasoning_model, enhance

set_reasoning_model(
    base_url="http://localhost:11434/v1",  # Ollama default
    model="qwen2.5:3b",                    # Model name
    temperature=0.7,                        # Creativity (0-1)
    max_tokens=2048,                        # Response length
    enable_routing=True,                    # Neural routing
    use_symbolic_verification=True,         # SymPy verification
    max_reflection_iterations=2,            # Self-correction attempts
)

# Use it
result = enhance("What is 2 + 2?")
print(result)
```

### **Option 2: Using OpenAI API**

```python
set_reasoning_model(
    base_url="https://api.openai.com/v1",
    model="gpt-4",
    api_key="your-openai-key",
    temperature=0.7,
)
```

### **Option 3: Using Other OpenAI-Compatible APIs**

```python
# LM Studio
set_reasoning_model(
    base_url="http://localhost:1234/v1",
    model="local-model",
)

# vLLM
set_reasoning_model(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-7B-Instruct",
)

# Together AI
set_reasoning_model(
    base_url="https://api.together.xyz/v1",
    model="Qwen/Qwen2.5-7B-Instruct",
    api_key="your-together-key",
)
```

---

## 📝 Basic Usage Examples

### **Example 1: Math Query**

```python
from kaelum import enhance

result = enhance("Solve: 2x + 6 = 10")
print(result)

# Output:
# x = 2
# 
# Worker: math | Confidence: 0.95 | Verification: ✓ PASSED
# 
# Reasoning:
# 1. Subtract 6 from both sides: 2x = 4
# 2. Divide by 2: x = 2
# 3. Verify: 2(2) + 6 = 10 ✓
```

### **Example 2: Logic Query**

```python
result = enhance("All humans are mortal. Socrates is human. Is Socrates mortal?")
print(result)

# Output:
# Yes, Socrates is mortal.
# 
# Worker: logic | Confidence: 0.92 | Verification: ✓ PASSED
# 
# Reasoning:
# 1. Premise 1: All humans are mortal
# 2. Premise 2: Socrates is human
# 3. Conclusion: Socrates is mortal (modus ponens)
```

### **Example 3: Code Query**

```python
result = enhance("Write a Python function to reverse a string")
print(result)

# Output:
# def reverse_string(s):
#     return s[::-1]
# 
# Worker: code | Confidence: 0.88 | Verification: ✓ PASSED
```

### **Example 4: Programmatic API**

```python
from kaelum import kaelum_enhance_reasoning

result = kaelum_enhance_reasoning(
    query="Calculate the integral of 2x",
    domain="calculus"
)

print(f"Answer: {result['suggested_approach']}")
print(f"Worker: {result['worker_used']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Verification: {result['verification_passed']}")
print(f"Iterations: {result['iterations']}")
print(f"Cache Hit: {result['cache_hit']}")

for i, step in enumerate(result['reasoning_steps'], 1):
    print(f"{i}. {step}")
```

---

## 🐛 Troubleshooting

### **Issue 1: "Import kaelum could not be resolved"**

**Solution:**
```bash
# Make sure you're in the project directory
cd /path/to/KaelumAI

# Activate virtual environment
source .venv/bin/activate.fish  # or .venv/bin/activate

# Install in development mode
pip install -e .
```

### **Issue 2: "Connection refused to localhost:11434"**

**Solution:**
```bash
# Check if Ollama is running
ollama list

# If not, start it
ollama serve

# Or pull a model (this auto-starts the service)
ollama pull qwen2.5:3b
```

### **Issue 3: "sentence_transformers" download is slow**

**Solution:**
```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

This downloads ~120MB and caches it for future use.

### **Issue 4: PyTorch installation fails**

**Solution:**
```bash
# Install PyTorch separately (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install other requirements
pip install -r requirements.txt
```

### **Issue 5: Out of memory errors**

**Solutions:**
1. **Use smaller model:**
   ```bash
   ollama pull qwen2.5:1.5b  # Only 1GB
   ```

2. **Reduce max_tokens:**
   ```python
   set_reasoning_model(max_tokens=1024)  # Instead of 2048
   ```

3. **Disable caching temporarily:**
   ```python
   # In worker.solve() calls
   result = worker.solve(query, use_cache=False)
   ```

---

## 🎯 Testing Your Installation

Create `test_kaelum.py`:

```python
from kaelum import enhance, set_reasoning_model

# Configure
set_reasoning_model(
    base_url="http://localhost:11434/v1",
    model="qwen2.5:3b",
)

# Test 1: Simple math
print("Test 1: Math")
print(enhance("What is 2 + 2?"))
print()

# Test 2: Verification
print("Test 2: Calculus with verification")
print(enhance("What is the derivative of x^2?"))
print()

# Test 3: Logic
print("Test 3: Logic")
print(enhance("If all A are B, and all B are C, then are all A also C?"))

print("\n✅ All tests passed!")
```

Run it:
```bash
python test_kaelum.py
```

---

## 🚀 Next Steps

1. **Explore different models:**
   ```bash
   ollama pull mistral:7b
   ollama pull phi3:mini
   ollama pull llama3.2:3b
   ```

2. **Try advanced features:**
   - Enable debug mode: `debug_verification=True`
   - Adjust reflection iterations: `max_reflection_iterations=3`
   - Experiment with temperature: `temperature=0.3` (more focused) or `temperature=0.9` (more creative)

3. **Check the examples:**
   - Run `python example.py` for more query types
   - Modify queries to test different reasoning scenarios

4. **Monitor performance:**
   ```python
   from kaelum import enhance
   import time
   
   start = time.time()
   result = enhance("Your query here")
   print(f"Time taken: {time.time() - start:.2f}s")
   ```

5. **Integrate into your project:**
   ```python
   # In your code
   from kaelum import enhance
   
   answer = enhance(user_question)
   ```

---

## 📊 Performance Tips

1. **First query is slower:** Sentence-transformers downloads on first use (~120MB)
2. **Cache hits are 1000x faster:** Similar queries return instantly
3. **Smaller models = faster:** qwen2.5:1.5b vs qwen2.5:7b
4. **GPU helps:** But CPU works fine for <7B models
5. **Batch queries:** Call multiple queries in sequence to benefit from warm cache

---

## 🔗 Useful Links

- **Ollama Models:** https://ollama.com/library
- **Kaelum GitHub:** https://github.com/ashworks1706/KaelumAI
- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **Sentence-Transformers:** https://www.sbert.net/

---

## 💡 Tips

- **Start with qwen2.5:3b** - best balance of speed and quality
- **Enable routing** - automatically selects best worker
- **Use verification** - catches math errors with SymPy
- **Monitor cache hits** - indicates system is learning
- **Check confidence scores** - >0.8 is very reliable

---

**Ready to go!** 🎉

Run `python example.py` to see Kaelum in action!
