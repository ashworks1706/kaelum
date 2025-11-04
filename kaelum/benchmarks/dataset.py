"""Benchmark dataset with diverse query types."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Any
import json


class QueryCategory(Enum):
    """Categories of benchmark queries."""
    MATH = "math"
    LOGIC = "logic"
    CODE = "code"
    FACTUAL = "factual"
    CREATIVE = "creative"
    MIXED = "mixed"


class DifficultyLevel(Enum):
    """Difficulty levels for queries."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class BenchmarkQuery:
    """A single benchmark query.
    
    Attributes:
        id: Unique identifier
        query: The question/prompt
        category: Query category
        difficulty: Difficulty level
        expected_answer: Expected answer (for evaluation)
        metadata: Additional information
    """
    id: str
    query: str
    category: QueryCategory
    difficulty: DifficultyLevel
    expected_answer: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "expected_answer": self.expected_answer,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkQuery":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            query=data["query"],
            category=QueryCategory(data["category"]),
            difficulty=DifficultyLevel(data["difficulty"]),
            expected_answer=data.get("expected_answer"),
            metadata=data.get("metadata", {})
        )


class BenchmarkDataset:
    """Collection of benchmark queries."""
    
    def __init__(self, name: str = "default"):
        """Initialize dataset.
        
        Args:
            name: Dataset name
        """
        self.name = name
        self.queries: List[BenchmarkQuery] = []
    
    def add_query(self, query: BenchmarkQuery):
        """Add a query to dataset."""
        self.queries.append(query)
    
    def add_queries(self, queries: List[BenchmarkQuery]):
        """Add multiple queries."""
        self.queries.extend(queries)
    
    def filter_by_category(self, category: QueryCategory) -> List[BenchmarkQuery]:
        """Get queries by category."""
        return [q for q in self.queries if q.category == category]
    
    def filter_by_difficulty(self, difficulty: DifficultyLevel) -> List[BenchmarkQuery]:
        """Get queries by difficulty."""
        return [q for q in self.queries if q.difficulty == difficulty]
    
    def get_by_id(self, query_id: str) -> Optional[BenchmarkQuery]:
        """Get query by ID."""
        for query in self.queries:
            if query.id == query_id:
                return query
        return None
    
    def save(self, filepath: str):
        """Save dataset to JSON file."""
        data = {
            "name": self.name,
            "queries": [q.to_dict() for q in self.queries]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "BenchmarkDataset":
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        dataset = cls(name=data["name"])
        dataset.queries = [
            BenchmarkQuery.from_dict(q) for q in data["queries"]
        ]
        return dataset
    
    def __len__(self) -> int:
        """Get number of queries."""
        return len(self.queries)
    
    def __iter__(self):
        """Iterate over queries."""
        return iter(self.queries)


def create_default_dataset() -> BenchmarkDataset:
    """Create default benchmark dataset with 100+ diverse queries."""
    dataset = BenchmarkDataset(name="kaelum_v1")
    
    # ==================== MATH QUERIES (25) ====================
    math_queries = [
        # Easy arithmetic (5)
        BenchmarkQuery("math_001", "Calculate 15 + 27", QueryCategory.MATH, DifficultyLevel.EASY, "42"),
        BenchmarkQuery("math_002", "What is 100 - 37?", QueryCategory.MATH, DifficultyLevel.EASY, "63"),
        BenchmarkQuery("math_003", "Multiply 8 by 7", QueryCategory.MATH, DifficultyLevel.EASY, "56"),
        BenchmarkQuery("math_004", "Divide 144 by 12", QueryCategory.MATH, DifficultyLevel.EASY, "12"),
        BenchmarkQuery("math_005", "What is 50% of 200?", QueryCategory.MATH, DifficultyLevel.EASY, "100"),
        
        # Medium algebra (10)
        BenchmarkQuery("math_006", "Solve for x: 2x + 5 = 15", QueryCategory.MATH, DifficultyLevel.MEDIUM, "5"),
        BenchmarkQuery("math_007", "Solve for y: 3y - 7 = 20", QueryCategory.MATH, DifficultyLevel.MEDIUM, "9"),
        BenchmarkQuery("math_008", "What is x if x^2 = 25?", QueryCategory.MATH, DifficultyLevel.MEDIUM, "5 or -5"),
        BenchmarkQuery("math_009", "Solve: 5x + 3 = 2x + 12", QueryCategory.MATH, DifficultyLevel.MEDIUM, "3"),
        BenchmarkQuery("math_010", "Factor: x^2 + 5x + 6", QueryCategory.MATH, DifficultyLevel.MEDIUM, "(x+2)(x+3)"),
        BenchmarkQuery("math_011", "Simplify: (x^2 - 4)/(x - 2)", QueryCategory.MATH, DifficultyLevel.MEDIUM, "x + 2"),
        BenchmarkQuery("math_012", "Expand: (x + 3)^2", QueryCategory.MATH, DifficultyLevel.MEDIUM, "x^2 + 6x + 9"),
        BenchmarkQuery("math_013", "Solve: 2x/3 = 8", QueryCategory.MATH, DifficultyLevel.MEDIUM, "12"),
        BenchmarkQuery("math_014", "What is the slope between (1,2) and (3,6)?", QueryCategory.MATH, DifficultyLevel.MEDIUM, "2"),
        BenchmarkQuery("math_015", "Solve the system: x + y = 10, x - y = 2", QueryCategory.MATH, DifficultyLevel.MEDIUM, "x=6, y=4"),
        
        # Hard calculus (10)
        BenchmarkQuery("math_016", "What is the derivative of x^2?", QueryCategory.MATH, DifficultyLevel.HARD, "2x"),
        BenchmarkQuery("math_017", "Find d/dx of x^3 + 2x", QueryCategory.MATH, DifficultyLevel.HARD, "3x^2 + 2"),
        BenchmarkQuery("math_018", "Integrate x^2 with respect to x", QueryCategory.MATH, DifficultyLevel.HARD, "x^3/3 + C"),
        BenchmarkQuery("math_019", "What is the derivative of sin(x)?", QueryCategory.MATH, DifficultyLevel.HARD, "cos(x)"),
        BenchmarkQuery("math_020", "Find the limit of (x^2 - 1)/(x - 1) as x approaches 1", QueryCategory.MATH, DifficultyLevel.HARD, "2"),
        BenchmarkQuery("math_021", "Differentiate e^x", QueryCategory.MATH, DifficultyLevel.HARD, "e^x"),
        BenchmarkQuery("math_022", "What is the integral of 1/x?", QueryCategory.MATH, DifficultyLevel.HARD, "ln|x| + C"),
        BenchmarkQuery("math_023", "Find d/dx of ln(x)", QueryCategory.MATH, DifficultyLevel.HARD, "1/x"),
        BenchmarkQuery("math_024", "What is the derivative of x^n?", QueryCategory.MATH, DifficultyLevel.HARD, "n*x^(n-1)"),
        BenchmarkQuery("math_025", "Evaluate the integral of 2x from 0 to 5", QueryCategory.MATH, DifficultyLevel.HARD, "25"),
    ]
    
    # ==================== LOGIC QUERIES (25) ====================
    logic_queries = [
        # Easy logic (8)
        BenchmarkQuery("logic_001", "If A is true and B is true, is (A AND B) true?", QueryCategory.LOGIC, DifficultyLevel.EASY, "true"),
        BenchmarkQuery("logic_002", "If P implies Q, and P is true, what can we conclude about Q?", QueryCategory.LOGIC, DifficultyLevel.EASY, "Q is true"),
        BenchmarkQuery("logic_003", "Is 'NOT NOT A' equivalent to A?", QueryCategory.LOGIC, DifficultyLevel.EASY, "yes"),
        BenchmarkQuery("logic_004", "If A OR B is true, and A is false, what must B be?", QueryCategory.LOGIC, DifficultyLevel.EASY, "true"),
        BenchmarkQuery("logic_005", "If all humans are mortal, is Socrates mortal?", QueryCategory.LOGIC, DifficultyLevel.EASY, "yes (if Socrates is human)"),
        BenchmarkQuery("logic_006", "True or False: (A AND NOT A) is always false", QueryCategory.LOGIC, DifficultyLevel.EASY, "true"),
        BenchmarkQuery("logic_007", "If it's raining, then the ground is wet. The ground is wet. Is it raining?", QueryCategory.LOGIC, DifficultyLevel.EASY, "not necessarily"),
        BenchmarkQuery("logic_008", "Can both 'A' and 'NOT A' be true simultaneously?", QueryCategory.LOGIC, DifficultyLevel.EASY, "no"),
        
        # Medium syllogisms (9)
        BenchmarkQuery("logic_009", "All humans are mortal. Socrates is human. Therefore?", QueryCategory.LOGIC, DifficultyLevel.MEDIUM, "Socrates is mortal"),
        BenchmarkQuery("logic_010", "All birds can fly. Penguins are birds. Can penguins fly?", QueryCategory.LOGIC, DifficultyLevel.MEDIUM, "The premise is false"),
        BenchmarkQuery("logic_011", "No cats are dogs. Some pets are cats. Are any pets dogs?", QueryCategory.LOGIC, DifficultyLevel.MEDIUM, "Cannot determine"),
        BenchmarkQuery("logic_012", "All A are B. All B are C. Therefore all A are?", QueryCategory.LOGIC, DifficultyLevel.MEDIUM, "C"),
        BenchmarkQuery("logic_013", "Some X are Y. All Y are Z. What can we conclude about X and Z?", QueryCategory.LOGIC, DifficultyLevel.MEDIUM, "Some X are Z"),
        BenchmarkQuery("logic_014", "If it rains, the game is cancelled. The game was played. Did it rain?", QueryCategory.LOGIC, DifficultyLevel.MEDIUM, "no"),
        BenchmarkQuery("logic_015", "All mammals have hair. Whales are mammals. Do whales have hair?", QueryCategory.LOGIC, DifficultyLevel.MEDIUM, "yes"),
        BenchmarkQuery("logic_016", "If A then B. If B then C. If A then?", QueryCategory.LOGIC, DifficultyLevel.MEDIUM, "C"),
        BenchmarkQuery("logic_017", "Either P or Q. Not P. Therefore?", QueryCategory.LOGIC, DifficultyLevel.MEDIUM, "Q"),
        
        # Hard proofs (8)
        BenchmarkQuery("logic_018", "Prove: If A implies B, and B implies C, then A implies C", QueryCategory.LOGIC, DifficultyLevel.HARD),
        BenchmarkQuery("logic_019", "Is modus tollens valid: If P then Q, NOT Q, therefore NOT P?", QueryCategory.LOGIC, DifficultyLevel.HARD, "yes"),
        BenchmarkQuery("logic_020", "Prove by contradiction: If n^2 is even, then n is even", QueryCategory.LOGIC, DifficultyLevel.HARD),
        BenchmarkQuery("logic_021", "Show that (A OR B) AND NOT A implies B", QueryCategory.LOGIC, DifficultyLevel.HARD),
        BenchmarkQuery("logic_022", "Is this valid: If P then Q, Q therefore P?", QueryCategory.LOGIC, DifficultyLevel.HARD, "no (affirming the consequent)"),
        BenchmarkQuery("logic_023", "Prove De Morgan's law: NOT(A AND B) = (NOT A) OR (NOT B)", QueryCategory.LOGIC, DifficultyLevel.HARD),
        BenchmarkQuery("logic_024", "If exactly one of A or B is true (XOR), and A is true, what is B?", QueryCategory.LOGIC, DifficultyLevel.HARD, "false"),
        BenchmarkQuery("logic_025", "Show that (A implies B) is equivalent to (NOT A OR B)", QueryCategory.LOGIC, DifficultyLevel.HARD),
    ]
    
    # ==================== CODE QUERIES (25) ====================
    code_queries = [
        # Easy syntax (8)
        BenchmarkQuery("code_001", "Write a Python function to add two numbers", QueryCategory.CODE, DifficultyLevel.EASY),
        BenchmarkQuery("code_002", "How do you reverse a string in Python?", QueryCategory.CODE, DifficultyLevel.EASY),
        BenchmarkQuery("code_003", "Write a for loop that prints 1 to 10", QueryCategory.CODE, DifficultyLevel.EASY),
        BenchmarkQuery("code_004", "What is the output of: print(type([1,2,3]))?", QueryCategory.CODE, DifficultyLevel.EASY, "<class 'list'>"),
        BenchmarkQuery("code_005", "Write a function to check if a number is even", QueryCategory.CODE, DifficultyLevel.EASY),
        BenchmarkQuery("code_006", "How do you create an empty dictionary in Python?", QueryCategory.CODE, DifficultyLevel.EASY, "{} or dict()"),
        BenchmarkQuery("code_007", "What does len([1,2,3,4,5]) return?", QueryCategory.CODE, DifficultyLevel.EASY, "5"),
        BenchmarkQuery("code_008", "Write a Python list comprehension to square numbers 1-5", QueryCategory.CODE, DifficultyLevel.EASY),
        
        # Medium algorithms (9)
        BenchmarkQuery("code_009", "Write a function to find the maximum element in a list", QueryCategory.CODE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("code_010", "Implement binary search in Python", QueryCategory.CODE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("code_011", "Write a function to check if a string is a palindrome", QueryCategory.CODE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("code_012", "Implement bubble sort", QueryCategory.CODE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("code_013", "Write a function to count word frequency in a string", QueryCategory.CODE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("code_014", "How do you remove duplicates from a list?", QueryCategory.CODE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("code_015", "Write a function to flatten a nested list", QueryCategory.CODE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("code_016", "Implement a simple stack class with push and pop", QueryCategory.CODE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("code_017", "Write a function to find the nth Fibonacci number", QueryCategory.CODE, DifficultyLevel.MEDIUM),
        
        # Hard algorithms (8)
        BenchmarkQuery("code_018", "Implement quicksort algorithm", QueryCategory.CODE, DifficultyLevel.HARD),
        BenchmarkQuery("code_019", "Write a function to detect a cycle in a linked list", QueryCategory.CODE, DifficultyLevel.HARD),
        BenchmarkQuery("code_020", "Implement depth-first search for a graph", QueryCategory.CODE, DifficultyLevel.HARD),
        BenchmarkQuery("code_021", "Solve: Find longest increasing subsequence", QueryCategory.CODE, DifficultyLevel.HARD),
        BenchmarkQuery("code_022", "Implement a LRU cache", QueryCategory.CODE, DifficultyLevel.HARD),
        BenchmarkQuery("code_023", "Write a function to solve the knapsack problem", QueryCategory.CODE, DifficultyLevel.HARD),
        BenchmarkQuery("code_024", "Implement Dijkstra's shortest path algorithm", QueryCategory.CODE, DifficultyLevel.HARD),
        BenchmarkQuery("code_025", "Write a regex to validate email addresses", QueryCategory.CODE, DifficultyLevel.HARD),
    ]
    
    # ==================== FACTUAL QUERIES (15) ====================
    factual_queries = [
        BenchmarkQuery("fact_001", "What is the capital of France?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "Paris"),
        BenchmarkQuery("fact_002", "Who wrote Romeo and Juliet?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "William Shakespeare"),
        BenchmarkQuery("fact_003", "What year did World War 2 end?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "1945"),
        BenchmarkQuery("fact_004", "What is the largest planet in our solar system?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "Jupiter"),
        BenchmarkQuery("fact_005", "Who painted the Mona Lisa?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "Leonardo da Vinci"),
        BenchmarkQuery("fact_006", "What is the speed of light?", QueryCategory.FACTUAL, DifficultyLevel.MEDIUM, "299,792,458 m/s"),
        BenchmarkQuery("fact_007", "What is the chemical formula for water?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "H2O"),
        BenchmarkQuery("fact_008", "Who was the first person to walk on the moon?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "Neil Armstrong"),
        BenchmarkQuery("fact_009", "What is the smallest prime number?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "2"),
        BenchmarkQuery("fact_010", "How many continents are there?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "7"),
        BenchmarkQuery("fact_011", "What is the boiling point of water in Celsius?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "100°C"),
        BenchmarkQuery("fact_012", "Who developed the theory of relativity?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "Albert Einstein"),
        BenchmarkQuery("fact_013", "What is the longest river in the world?", QueryCategory.FACTUAL, DifficultyLevel.MEDIUM, "Nile or Amazon (disputed)"),
        BenchmarkQuery("fact_014", "What is the powerhouse of the cell?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "Mitochondria"),
        BenchmarkQuery("fact_015", "When was the Declaration of Independence signed?", QueryCategory.FACTUAL, DifficultyLevel.EASY, "1776"),
    ]
    
    # ==================== CREATIVE QUERIES (10) ====================
    creative_queries = [
        BenchmarkQuery("creative_001", "Write a haiku about coding", QueryCategory.CREATIVE, DifficultyLevel.EASY),
        BenchmarkQuery("creative_002", "Generate 3 creative names for a coffee shop", QueryCategory.CREATIVE, DifficultyLevel.EASY),
        BenchmarkQuery("creative_003", "Write a short story opening in one sentence", QueryCategory.CREATIVE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("creative_004", "Create a metaphor comparing life to a journey", QueryCategory.CREATIVE, DifficultyLevel.EASY),
        BenchmarkQuery("creative_005", "Brainstorm 5 ideas for a mobile app", QueryCategory.CREATIVE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("creative_006", "Write a limerick about artificial intelligence", QueryCategory.CREATIVE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("creative_007", "Describe the color blue without using color words", QueryCategory.CREATIVE, DifficultyLevel.HARD),
        BenchmarkQuery("creative_008", "Create an analogy for how neural networks work", QueryCategory.CREATIVE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("creative_009", "Write a dialogue between two robots", QueryCategory.CREATIVE, DifficultyLevel.MEDIUM),
        BenchmarkQuery("creative_010", "Generate a creative solution to reduce plastic waste", QueryCategory.CREATIVE, DifficultyLevel.HARD),
    ]
    
    # Add all queries
    dataset.add_queries(math_queries)
    dataset.add_queries(logic_queries)
    dataset.add_queries(code_queries)
    dataset.add_queries(factual_queries)
    dataset.add_queries(creative_queries)
    
    return dataset
