"""KaelumAI - Stress Test Demo

Start vLLM server first (for 6GB GPU with AWQ quantization):
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --port 8000 \
        --quantization awq \
        --gpu-memory-utilization 0.9 \
        --max-model-len 2048

For testing different models, change the model parameter below.
"""
from kaelum import enhance_stream, set_reasoning_model


# Test cases to stress test Kaelum's reasoning + verification + reflection
TEST_CASES = {
    "1_math_verification": """
A store is having a sale. A laptop originally costs $899. 
First, there's a 15% discount. Then, an additional $50 coupon is applied.
Finally, 8.5% sales tax is added to the discounted price.

What is the final price the customer pays? Show all calculations.
""",

    "2_logic_puzzle": """
There are three boxes: one contains only apples, one contains only oranges, 
and one contains both apples and oranges. All boxes are labeled incorrectly.

You can pick one fruit from one box to determine the contents of all boxes.
Which box should you pick from, and how does this tell you all contents?

Explain your reasoning step-by-step.
""",

    "3_financial_planning": """
Sarah earns $4,500/month after taxes. Her expenses are:
- Rent: $1,200
- Food: $400
- Transportation: $250
- Utilities: $150
- Insurance: $200

She wants to save 20% of her remaining income and invest the rest in:
- 60% stocks (7% annual return)
- 40% bonds (3% annual return)

How much will she have after 5 years if she follows this plan?
Assume monthly compounding and she maintains this strategy.
""",

    "4_code_debugging": """
Here's a Python function that's supposed to find the second largest number in a list:

def second_largest(arr):
    arr.sort()
    return arr[-2]

Test cases:
1. [5, 2, 8, 1, 9] → Expected: 8
2. [10, 10, 9, 8] → Expected: 9
3. [7] → Expected: Error
4. [5, 5, 5, 5] → Expected: 5

Identify what's wrong with this function and explain why each test case 
passes or fails. Then provide the corrected version.
""",

    "5_physics": """
A ball is thrown upward from the ground with an initial velocity of 20 m/s.
Acceleration due to gravity is -9.8 m/s².

Calculate:
1. Maximum height reached
2. Time to reach maximum height
3. Total time in the air
4. Velocity when it hits the ground

Show all formulas and work.
""",

    "6_probability": """
A factory produces widgets. Quality control shows:
- 95% of widgets pass inspection
- Of passing widgets, 2% fail in the first year
- Of failing widgets, 40% fail in the first year

A customer buys a widget that fails in the first year.
What is the probability it originally passed quality control?

Use Bayes' theorem and show all steps.
""",

    "7_word_problem": """
A train travels from City A to City B at 60 mph. Another train travels from 
City B to City A at 80 mph. The cities are 420 miles apart. They start at 
the same time.

When they meet:
1. How far has each train traveled?
2. How long have they been traveling?
3. How far are they from City A?

After they meet, the first train increases speed to 75 mph. 
4. How much longer until it reaches City B?

Show all work and explain your reasoning.
"""
}


# Select test case (change this to test different scenarios)
ACTIVE_TEST = "1_math_verification"  # Change to any key from TEST_CASES
query = TEST_CASES[ACTIVE_TEST].strip()


# Reasoning system prompt - optimized for step-by-step verification
reasoning_system = """You are an expert reasoning assistant. For every problem:

1. **Understand**: Clearly state what is being asked
2. **Identify**: List all known information and constraints
3. **Plan**: Break down the solution into logical steps
4. **Execute**: Work through each step carefully, showing all work
5. **Verify**: Check your calculations and logic at each step

For mathematical problems:
- Write out all formulas before substituting values
- Show every calculation step
- Include units in your answers

For logical problems:
- State assumptions explicitly
- Explain why each step follows from the previous
- Consider edge cases

Present your reasoning as a clear, numbered list."""

# User template - emphasizes careful step-by-step thinking
reasoning_template = "Solve this problem step-by-step, showing all your work:\n\n{query}"


# Configure Kaelum
set_reasoning_model(
    base_url="http://localhost:11434/v1",  # Ollama default
    model="qwen3:4b",                        # Or your chosen model
    use_symbolic_verification=True,
    use_factual_verification=False,
    debug_verification=True  # Enable debug logging for math verification
)


print(f"{'='*70}")
print(f"Test Case: {ACTIVE_TEST}")
print(f"{'='*70}\n")
print(f"Query:\n{query}\n")
print(f"{'='*70}\n")

try:
    for chunk in enhance_stream(query):
        print(chunk, end='', flush=True)
    print()
except Exception as e:
    print(f"\n❌ Error: {e}")

