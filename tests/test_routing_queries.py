"""Comprehensive test queries for routing classification.

This module contains 100+ diverse queries across different categories
to test and improve routing accuracy.
"""

import pytest
from kaelum.core.router import Router, QueryType, ReasoningStrategy


# Test queries organized by expected type
MATH_QUERIES = [
    # Basic arithmetic
    "What is 15 + 27?",
    "Calculate 144 divided by 12",
    "What's 8 times 9?",
    "Subtract 45 from 100",
    
    # Word problems
    "If John has 5 apples and buys 3 more, how many does he have?",
    "A train travels 60 mph for 2.5 hours. How far does it go?",
    "Sarah spent $45 on groceries and $23 on gas. What's the total?",
    
    # Algebra
    "Solve for x: 2x + 5 = 15",
    "What is x when 3x - 7 = 20?",
    "Simplify: (x^2 + 2x + 1)",
    
    # Calculus
    "What is the derivative of x^2 + 3x?",
    "Find the integral of 2x dx",
    "Calculate the limit as x approaches 0 of sin(x)/x",
    
    # Geometry
    "What's the area of a circle with radius 5?",
    "Find the volume of a sphere with radius 3",
    "What's the circumference of a circle with diameter 10?",
    
    # Statistics
    "What's the mean of [2, 4, 6, 8, 10]?",
    "Calculate the standard deviation of [1, 2, 3, 4, 5]",
    "What's the median of [5, 2, 8, 1, 9]?",
    
    # Complex math
    "Solve the quadratic equation: x^2 - 5x + 6 = 0",
]

LOGIC_QUERIES = [
    # Deductive reasoning
    "If all cats are animals, and Fluffy is a cat, is Fluffy an animal?",
    "If A implies B, and B implies C, does A imply C?",
    "If it's raining, the ground is wet. The ground is wet. Is it raining?",
    
    # Logical puzzles
    "In a room with red, blue, and green boxes, if the red box is bigger than blue, and blue is bigger than green, which is smallest?",
    "If Alice is taller than Bob, and Bob is taller than Carol, who is shortest?",
    "Three people: one always tells truth, one always lies, one random. How to identify the truth-teller?",
    
    # Syllogisms
    "All humans are mortal. Socrates is human. Therefore?",
    "No fish are mammals. All dolphins are mammals. Therefore?",
    "Some birds can fly. Penguins are birds. Can penguins fly?",
    
    # Propositional logic
    "Is (A AND B) OR C equivalent to (A OR C) AND (B OR C)?",
    "What's the negation of 'All swans are white'?",
    "If P implies Q, what can we conclude from NOT Q?",
    
    # Logical fallacies
    "Is this valid: 'If it rains, streets are wet. Streets are wet. Therefore it rained.'?",
    "Identify the fallacy: 'Everyone believes X, therefore X is true'",
    
    # Proof-based
    "Prove that if n is even, then n^2 is even",
    "Show that the square root of 2 is irrational",
    "Prove by contradiction that there are infinite primes",
    
    # Set theory
    "What's the intersection of {1,2,3} and {2,3,4}?",
    "Is the empty set a subset of every set?",
    "What's the power set of {a, b}?",
]

CODE_QUERIES = [
    # Algorithm questions
    "Write a function to reverse a string",
    "Implement binary search in Python",
    "Code a function to check if a number is prime",
    "Write an algorithm to sort an array",
    
    # Debugging
    "Why does this code give IndexError: list index out of range?",
    "Fix this Python syntax error: def foo( print('hello')",
    "Debug: why is my loop infinite?",
    "What's wrong with this recursive function?",
    
    # Data structures
    "Implement a stack using Python lists",
    "How to create a binary tree in Python?",
    "Write a linked list class",
    "Implement a hash map",
    
    # Complexity analysis
    "What's the time complexity of bubble sort?",
    "Analyze the space complexity of quicksort",
    "Is this algorithm O(n) or O(n^2)?",
    
    # Language-specific
    "What's the difference between == and === in JavaScript?",
    "Explain list comprehension in Python",
    "How do Python decorators work?",
    "What are async/await in JavaScript?",
    
    # System design
    "Design a URL shortener service",
    "How would you implement a cache?",
    "Explain the architecture of a REST API",
]

FACTUAL_QUERIES = [
    # History
    "When did World War 2 end?",
    "Who was the first president of the United States?",
    "What year did the Berlin Wall fall?",
    
    # Geography
    "What's the capital of France?",
    "Which is the largest ocean?",
    "Name the seven continents",
    
    # Science
    "What's the speed of light?",
    "How many planets are in the solar system?",
    "What's the chemical formula for water?",
    
    # General knowledge
    "Who wrote Romeo and Juliet?",
    "What's the tallest mountain in the world?",
    "How many bones are in the human body?",
    
    # Current events (time-sensitive)
    "Who is the current CEO of Tesla?",
    "What's the latest version of Python?",
    "When was ChatGPT released?",
    
    # Definitions
    "What is machine learning?",
    "Define quantum computing",
    "Explain what DNA is",
    
    # Statistics/Facts
    "What's the population of Earth?",
    "How tall is the Eiffel Tower?",
    "What's the boiling point of water?",
]

CREATIVE_QUERIES = [
    # Creative writing
    "Write a haiku about autumn",
    "Create a short story about a time traveler",
    "Compose a poem about the ocean",
    
    # Brainstorming
    "Give me 5 creative names for a coffee shop",
    "Suggest unique gift ideas for a programmer",
    "What are some creative uses for old tires?",
    
    # Analogies and metaphors
    "Explain quantum mechanics using a cooking analogy",
    "Describe democracy like you're explaining to a 5-year-old",
    "Use a metaphor to explain how the internet works",
    
    # Hypotheticals
    "What would happen if gravity suddenly doubled?",
    "Imagine a world where everyone can read minds",
    "If you could have dinner with any historical figure, who and why?",
    
    # Storytelling
    "Tell me a story about a robot learning to feel emotions",
    "Create a fairy tale with a modern twist",
    "Write a dialogue between the sun and the moon",
    
    # Creative problem solving
    "How would you use a paperclip to escape a locked room?",
    "Invent a new sport that combines chess and soccer",
    "Design a house for someone who lives underwater",
]

ANALYSIS_QUERIES = [
    # Comparison
    "Compare and contrast democracy and monarchy",
    "What are the pros and cons of remote work?",
    "Analyze the differences between Python and JavaScript",
    
    # Interpretation
    "What does this quote mean: 'The only constant is change'?",
    "Interpret the symbolism in The Great Gatsby",
    "Analyze the themes in 1984 by George Orwell",
    
    # Evaluation
    "Evaluate the impact of social media on society",
    "Assess the effectiveness of renewable energy",
    "Critique the argument for universal basic income",
    
    # Cause and effect
    "What caused the 2008 financial crisis?",
    "Analyze the effects of climate change",
    "What are the consequences of deforestation?",
    
    # Trend analysis
    "Analyze the trend of AI adoption in healthcare",
    "What are the emerging patterns in remote education?",
    "Evaluate the growth trajectory of electric vehicles",
]

# Edge cases and ambiguous queries
EDGE_CASE_QUERIES = [
    # Multi-category (math + code)
    "Write Python code to calculate the fibonacci sequence",
    "Implement a function to solve quadratic equations",
    "Code a prime number generator using the Sieve of Eratosthenes",
    
    # Multi-category (logic + code)
    "Write a program to solve sudoku puzzles",
    "Implement a truth table generator in Python",
    "Code a logical expression parser",
    
    # Multi-category (factual + analysis)
    "Explain the causes and effects of the Industrial Revolution",
    "Analyze the historical significance of the printing press",
    "Evaluate the impact of the Renaissance on modern society",
    
    # Ambiguous/unclear
    "Help me understand this better",
    "What do you think about this?",
    "Can you explain?",
    "Tell me more",
    
    # Very short
    "Why?",
    "How?",
    "When?",
    
    # Multiple questions
    "What is 2+2 and what's the capital of France?",
    "Who invented the telephone and when did World War 1 start?",
]


class TestRoutingClassification:
    """Test routing classification accuracy."""
    
    @pytest.fixture
    def router(self):
        """Create a router instance for testing."""
        return Router(learning_enabled=False)
    
    def test_math_classification(self, router):
        """Test classification of math queries."""
        correct = 0
        total = len(MATH_QUERIES)
        
        for query in MATH_QUERIES:
            decision = router.route(query)
            if decision.query_type == QueryType.MATH:
                correct += 1
        
        accuracy = correct / total * 100
        print(f"\nMath classification accuracy: {accuracy:.1f}% ({correct}/{total})")
        assert accuracy >= 70, f"Math classification accuracy too low: {accuracy:.1f}%"
    
    def test_logic_classification(self, router):
        """Test classification of logic queries."""
        correct = 0
        total = len(LOGIC_QUERIES)
        
        for query in LOGIC_QUERIES:
            decision = router.route(query)
            if decision.query_type == QueryType.LOGIC:
                correct += 1
        
        accuracy = correct / total * 100
        print(f"\nLogic classification accuracy: {accuracy:.1f}% ({correct}/{total})")
        assert accuracy >= 60, f"Logic classification accuracy too low: {accuracy:.1f}%"
    
    def test_code_classification(self, router):
        """Test classification of code queries."""
        correct = 0
        total = len(CODE_QUERIES)
        
        for query in CODE_QUERIES:
            decision = router.route(query)
            if decision.query_type == QueryType.CODE:
                correct += 1
        
        accuracy = correct / total * 100
        print(f"\nCode classification accuracy: {accuracy:.1f}% ({correct}/{total})")
        assert accuracy >= 70, f"Code classification accuracy too low: {accuracy:.1f}%"
    
    def test_factual_classification(self, router):
        """Test classification of factual queries."""
        correct = 0
        total = len(FACTUAL_QUERIES)
        
        for query in FACTUAL_QUERIES:
            decision = router.route(query)
            if decision.query_type == QueryType.FACTUAL:
                correct += 1
        
        accuracy = correct / total * 100
        print(f"\nFactual classification accuracy: {accuracy:.1f}% ({correct}/{total})")
        assert accuracy >= 70, f"Factual classification accuracy too low: {accuracy:.1f}%"
    
    def test_creative_classification(self, router):
        """Test classification of creative queries."""
        correct = 0
        total = len(CREATIVE_QUERIES)
        
        for query in CREATIVE_QUERIES:
            decision = router.route(query)
            if decision.query_type == QueryType.CREATIVE:
                correct += 1
        
        accuracy = correct / total * 100
        print(f"\nCreative classification accuracy: {accuracy:.1f}% ({correct}/{total})")
        assert accuracy >= 60, f"Creative classification accuracy too low: {accuracy:.1f}%"
    
    def test_analysis_classification(self, router):
        """Test classification of analysis queries."""
        correct = 0
        total = len(ANALYSIS_QUERIES)
        
        for query in ANALYSIS_QUERIES:
            decision = router.route(query)
            if decision.query_type == QueryType.ANALYSIS:
                correct += 1
        
        accuracy = correct / total * 100
        print(f"\nAnalysis classification accuracy: {accuracy:.1f}% ({correct}/{total})")
        assert accuracy >= 60, f"Analysis classification accuracy too low: {accuracy:.1f}%"
    
    def test_overall_classification(self, router):
        """Test overall classification accuracy across all query types."""
        all_queries = [
            (MATH_QUERIES, QueryType.MATH),
            (LOGIC_QUERIES, QueryType.LOGIC),
            (CODE_QUERIES, QueryType.CODE),
            (FACTUAL_QUERIES, QueryType.FACTUAL),
            (CREATIVE_QUERIES, QueryType.CREATIVE),
            (ANALYSIS_QUERIES, QueryType.ANALYSIS),
        ]
        
        correct = 0
        total = 0
        
        for queries, expected_type in all_queries:
            for query in queries:
                total += 1
                decision = router.route(query)
                if decision.query_type == expected_type:
                    correct += 1
        
        accuracy = correct / total * 100
        print(f"\n" + "="*60)
        print(f"OVERALL CLASSIFICATION ACCURACY: {accuracy:.1f}% ({correct}/{total})")
        print("="*60)
        
        assert accuracy >= 65, f"Overall classification accuracy too low: {accuracy:.1f}%"
    
    def test_strategy_selection_consistency(self, router):
        """Test that strategy selection is consistent for similar queries."""
        # Test multiple times with same query
        query = "What is 2 + 2?"
        decisions = [router.route(query) for _ in range(5)]
        
        # All decisions should be the same
        strategies = [d.strategy for d in decisions]
        assert len(set(strategies)) == 1, "Strategy selection not consistent"
        
        query_types = [d.query_type for d in decisions]
        assert len(set(query_types)) == 1, "Query type classification not consistent"
    
    def test_edge_cases(self, router):
        """Test edge cases and ambiguous queries."""
        # These should not crash, but may classify as UNKNOWN
        for query in EDGE_CASE_QUERIES:
            try:
                decision = router.route(query)
                assert decision is not None
                assert decision.query_type in QueryType
                assert decision.strategy in ReasoningStrategy
            except Exception as e:
                pytest.fail(f"Router crashed on query '{query}': {e}")


# Utility function to export queries for manual review
def export_test_queries(filename="routing_test_queries.json"):
    """Export all test queries to JSON for manual review."""
    import json
    
    data = {
        "math": MATH_QUERIES,
        "logic": LOGIC_QUERIES,
        "code": CODE_QUERIES,
        "factual": FACTUAL_QUERIES,
        "creative": CREATIVE_QUERIES,
        "analysis": ANALYSIS_QUERIES,
        "edge_cases": EDGE_CASE_QUERIES,
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {sum(len(v) for v in data.values())} queries to {filename}")


if __name__ == "__main__":
    # Run tests or export queries
    export_test_queries()
