"""
Simple demo of KaelumAI - For serious testing, use test_notebooks/kaelum_testing.ipynb
"""

from kaelum import enhance

print("🧠 KaelumAI - Quick Demo")
print("=" * 60)
print("For customizable testing, open: test_notebooks/kaelum_testing.ipynb")
print("=" * 60)
print()

# Simple test
query = "What is 25% of 80?"
print(f"Query: {query}")
print()

result = enhance(query)
print(result)
print()
print("✅ Demo complete! Check test_notebooks/ for detailed experiments.")

