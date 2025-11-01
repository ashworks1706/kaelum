"""
Quick start example for KaelumAI - Tweak parameters here!
"""

import sys
from kaelum import enhance

# ============================================================================
# 🎛️  CUSTOMIZATION ZONE - Tweak these parameters!
# ============================================================================

# Model Selection
# Available: llama3.2:3b (fastest), qwen2.5:7b (better quality)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"

# Performance Presets
# Speed Mode: Fast responses, good for simple queries
SPEED_MODE = {
    "temperature": 0.3,      # Lower = more focused/deterministic
    "max_tokens": 512,       # Shorter responses
    "max_iterations": 1,     # Skip reflection for speed
}

# Quality Mode: Better reasoning, slower
QUALITY_MODE = {
    "temperature": 0.7,      # More creative/exploratory
    "max_tokens": 2048,      # Longer, detailed responses
    "max_iterations": 2,     # Enable reflection loop
}

# Balanced Mode (default)
BALANCED_MODE = {
    "temperature": 0.5,
    "max_tokens": 1024,
    "max_iterations": 1,
}

# 👇 Choose your preset here
CURRENT_MODE = SPEED_MODE  # Change to QUALITY_MODE or BALANCED_MODE

# ============================================================================

print(f"🚀 Model: {MODEL}")
print(f"⚙️  Mode: {list(CURRENT_MODE.values())} (temp={CURRENT_MODE['temperature']}, tokens={CURRENT_MODE['max_tokens']}, iter={CURRENT_MODE['max_iterations']})")
print(f"💡 Tip: Edit CURRENT_MODE in example.py to switch between SPEED/QUALITY/BALANCED\n")

# Example 1: Simple math question
print("=" * 60)
print("Example 1: Quick math")
print("=" * 60)

result = enhance("What is 25% of 80?", model=MODEL, **CURRENT_MODE)
print(f"Result: {result}")
print()

# Example 2: With mode specification
print("=" * 60)
print("Example 2: Equation solving")
print("=" * 60)

result = enhance("Solve: 3x + 5 = 20", mode="math", model=MODEL, **CURRENT_MODE)
print(f"Result: {result}")
print()

# Example 3: Complex reasoning
print("=" * 60)
print("Example 3: Logic reasoning")
print("=" * 60)

result = enhance(
    "If all birds can fly, and penguins are birds, can penguins fly?",
    mode="logic",
    model=MODEL,
    **CURRENT_MODE
)
print(f"Result: {result}")
print()

print("✅ All examples completed!")
print(f"\n🔧 To customize: Edit the CURRENT_MODE variable at the top of example.py")
print(f"🚀 To change model: python example.py qwen2.5:7b")

