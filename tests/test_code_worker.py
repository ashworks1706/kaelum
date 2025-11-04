"""Tests for CodeWorker."""

import pytest
import asyncio

from kaelum.core.code_worker import CodeWorker
from kaelum.core.workers import WorkerSpecialty


class MockLLMClient:
    """Mock LLM client for testing."""
    
    async def generate(self, prompt: str) -> str:
        """Generate mock code response."""
        prompt_lower = prompt.lower()
        
        # Check for specific tasks first (more specific matches)
        if "reverse" in prompt_lower and "string" in prompt_lower:
            return """Here's how to reverse a string in Python:

```python
def reverse_string(s: str) -> str:
    \"\"\"Reverse a string.\"\"\"
    return s[::-1]
```"""
        
        elif "debug" in prompt_lower or "fix" in prompt_lower:
            return """The issue is a missing colon. Fixed code:

```python
def calculate(x):
    if x > 0:  # Added missing colon
        return x * 2
    return 0
```"""
        
        elif ("algorithm" in prompt_lower and "sort" in prompt_lower) or "bubble sort" in prompt_lower:
            return """Here's a bubble sort implementation:

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```"""
        
        # Generic function generation (least specific, check last)
        elif "function" in prompt_lower or "add" in prompt_lower or "write" in prompt_lower:
            return """Here's a Python function to add two numbers:

```python
def add_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers and return the result.\"\"\"
    return a + b
```

This function takes two integers and returns their sum."""
        
        else:
            return "Mock code response without code block"


class MockConfig:
    """Mock configuration."""
    def __init__(self):
        self.reasoning_llm = {"model": "mock-model"}


# ==================== Basic Tests ====================

class TestCodeWorkerBasics:
    """Basic CodeWorker tests."""
    
    @pytest.fixture
    def worker(self, monkeypatch):
        """Create CodeWorker with mock LLM."""
        config = MockConfig()
        worker = CodeWorker.__new__(CodeWorker)
        worker.config = config
        worker.llm_client = MockLLMClient()
        worker.supported_languages = {
            'python', 'javascript', 'typescript', 'java', 'cpp', 'c',
            'go', 'rust', 'ruby', 'php', 'swift', 'kotlin'
        }
        return worker
    
    def test_get_specialty(self, worker):
        """Test specialty is CODE."""
        assert worker.get_specialty() == WorkerSpecialty.CODE
    
    def test_can_handle_code_keywords(self, worker):
        """Test recognition of code-related keywords."""
        queries = [
            "Write a Python function",
            "Implement an algorithm",
            "Debug this code",
            "Fix the syntax error",
            "Create a class for users"
        ]
        
        for query in queries:
            score = worker.can_handle(query)
            assert score > 0.5, f"Should handle: {query}"
    
    def test_can_handle_programming_languages(self, worker):
        """Test recognition of programming languages."""
        queries = [
            "Write a Python script",
            "Create a JavaScript function",
            "Implement in Java",
            "Use C++ for this",
        ]
        
        for query in queries:
            score = worker.can_handle(query)
            assert score > 0.5, f"Should handle: {query}"
    
    def test_can_handle_code_patterns(self, worker):
        """Test recognition of code patterns."""
        queries = [
            "def factorial(n):",
            "function calculateTotal() {",
            "class User:",
            "for item in list:",
            "if x > 0:",
        ]
        
        for query in queries:
            score = worker.can_handle(query)
            assert score > 0.3, f"Should recognize pattern: {query}"
    
    def test_cannot_handle_non_code(self, worker):
        """Test rejection of non-code queries."""
        queries = [
            "What is the capital of France?",
            "Solve x + 5 = 10",
            "Tell me a story",
        ]
        
        for query in queries:
            score = worker.can_handle(query)
            assert score < 0.5, f"Should not handle: {query}"


# ==================== Language Detection Tests ====================

class TestLanguageDetection:
    """Test programming language detection."""
    
    @pytest.fixture
    def worker(self):
        """Create CodeWorker."""
        config = MockConfig()
        worker = CodeWorker.__new__(CodeWorker)
        worker.config = config
        worker.llm_client = MockLLMClient()
        worker.supported_languages = {
            'python', 'javascript', 'typescript', 'java', 'cpp', 'c',
            'go', 'rust', 'ruby', 'php', 'swift', 'kotlin'
        }
        return worker
    
    def test_detect_python(self, worker):
        """Test Python detection."""
        queries = [
            "Write a Python function",
            "def my_function():",
            "import numpy",
            "script.py contains"
        ]
        
        for query in queries:
            lang = worker._detect_language(query)
            assert lang == 'python', f"Should detect Python: {query}"
    
    def test_detect_javascript(self, worker):
        """Test JavaScript detection."""
        queries = [
            "Write a JavaScript function",
            "function myFunc() {",
            "const x = 5;",
            "file.js contains"
        ]
        
        for query in queries:
            lang = worker._detect_language(query)
            assert lang == 'javascript', f"Should detect JavaScript: {query}"
    
    def test_detect_java(self, worker):
        """Test Java detection."""
        queries = [
            "Write Java code",
            "public class MyClass",
            "MyFile.java contains"
        ]
        
        for query in queries:
            lang = worker._detect_language(query)
            assert lang == 'java', f"Should detect Java: {query}"
    
    def test_detect_unspecified(self, worker):
        """Test detection when language unspecified."""
        query = "Write a function to add numbers"
        lang = worker._detect_language(query)
        # Could be None or inferred
        assert lang in [None, 'python', 'javascript']


# ==================== Task Classification Tests ====================

class TestTaskClassification:
    """Test coding task classification."""
    
    @pytest.fixture
    def worker(self):
        """Create CodeWorker."""
        config = MockConfig()
        worker = CodeWorker.__new__(CodeWorker)
        worker.config = config
        worker.llm_client = MockLLMClient()
        worker.supported_languages = {'python'}
        return worker
    
    def test_classify_debugging(self, worker):
        """Test debugging task classification."""
        queries = [
            "Debug this code",
            "Fix the error",
            "Find the bug",
        ]
        
        for query in queries:
            task_type = worker._classify_task(query)
            assert task_type == 'debugging'
    
    def test_classify_optimization(self, worker):
        """Test optimization task classification."""
        queries = [
            "Optimize this function",
            "Improve performance",
            "Refactor the code",
        ]
        
        for query in queries:
            task_type = worker._classify_task(query)
            assert task_type == 'optimization'
    
    def test_classify_review(self, worker):
        """Test review task classification."""
        queries = [
            "Review my code",
            "Analyze this function",
            "Explain what this does",
        ]
        
        for query in queries:
            task_type = worker._classify_task(query)
            assert task_type == 'review'
    
    def test_classify_testing(self, worker):
        """Test testing task classification."""
        queries = [
            "Write unit tests",
            "Create pytest tests",
            "Add test cases",
        ]
        
        for query in queries:
            task_type = worker._classify_task(query)
            assert task_type == 'testing'
    
    def test_classify_algorithm(self, worker):
        """Test algorithm task classification."""
        queries = [
            "Implement quicksort algorithm",
            "Create a binary tree data structure",
        ]
        
        for query in queries:
            task_type = worker._classify_task(query)
            assert task_type == 'algorithm'
    
    def test_classify_generation(self, worker):
        """Test generation as default."""
        query = "Write a function to add numbers"
        task_type = worker._classify_task(query)
        assert task_type == 'generation'


# ==================== Code Extraction Tests ====================

class TestCodeExtraction:
    """Test code extraction from responses."""
    
    @pytest.fixture
    def worker(self):
        """Create CodeWorker."""
        config = MockConfig()
        worker = CodeWorker.__new__(CodeWorker)
        worker.config = config
        worker.llm_client = MockLLMClient()
        worker.supported_languages = {'python'}
        return worker
    
    def test_extract_from_markdown(self, worker):
        """Test extraction from markdown code blocks."""
        response = """Here's the code:

```python
def add(a, b):
    return a + b
```

This function adds two numbers."""
        
        code = worker._extract_code(response)
        assert code is not None
        assert 'def add' in code
        assert 'return a + b' in code
    
    def test_extract_without_language(self, worker):
        """Test extraction from code blocks without language."""
        response = """Here's the solution:

```
function add(a, b) {
    return a + b;
}
```"""
        
        code = worker._extract_code(response)
        assert code is not None
        assert 'function add' in code
    
    def test_extract_no_code(self, worker):
        """Test when no code block present."""
        response = "This is just explanatory text without any code."
        
        code = worker._extract_code(response)
        # Should be None or empty
        assert code is None or code == ""


# ==================== Syntax Validation Tests ====================

class TestSyntaxValidation:
    """Test Python syntax validation."""
    
    @pytest.fixture
    def worker(self):
        """Create CodeWorker."""
        config = MockConfig()
        worker = CodeWorker.__new__(CodeWorker)
        worker.config = config
        worker.llm_client = MockLLMClient()
        worker.supported_languages = {'python'}
        return worker
    
    def test_valid_python_syntax(self, worker):
        """Test validation of valid Python code."""
        code = """def add(a, b):
    return a + b

result = add(2, 3)
print(result)"""
        
        assert worker._validate_python_syntax(code) is True
    
    def test_invalid_python_syntax(self, worker):
        """Test detection of invalid Python syntax."""
        code = """def add(a, b)  # Missing colon
    return a + b"""
        
        assert worker._validate_python_syntax(code) is False
    
    def test_invalid_indentation(self, worker):
        """Test detection of indentation errors."""
        code = """def add(a, b):
return a + b  # Wrong indentation"""
        
        assert worker._validate_python_syntax(code) is False


# ==================== Integration Tests ====================

class TestCodeWorkerIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def worker(self):
        """Create CodeWorker with mock LLM."""
        config = MockConfig()
        worker = CodeWorker.__new__(CodeWorker)
        worker.config = config
        worker.llm_client = MockLLMClient()
        worker.supported_languages = {
            'python', 'javascript', 'typescript', 'java', 'cpp', 'c',
            'go', 'rust', 'ruby', 'php', 'swift', 'kotlin'
        }
        return worker
    
    @pytest.mark.asyncio
    async def test_solve_simple_function(self, worker):
        """Test solving a simple function generation request."""
        query = "Write a Python function to add two numbers"
        result = await worker.solve(query)
        
        assert result.answer
        assert result.confidence > 0.5
        assert result.specialty == WorkerSpecialty.CODE
        assert result.metadata['language'] == 'python'
        assert result.metadata['task_type'] == 'generation'
    
    @pytest.mark.asyncio
    async def test_solve_string_reversal(self, worker):
        """Test solving string reversal."""
        query = "How do you reverse a string in Python?"
        result = await worker.solve(query)
        
        assert result.answer
        assert 'reverse' in result.answer.lower()
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_solve_debugging(self, worker):
        """Test debugging task."""
        query = "Debug this Python code: def calculate(x) if x > 0 return x * 2"
        result = await worker.solve(query)
        
        assert result.answer
        assert result.metadata['task_type'] == 'debugging'
        assert result.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_solve_algorithm(self, worker):
        """Test algorithm implementation."""
        query = "Implement bubble sort algorithm in Python"
        result = await worker.solve(query)
        
        assert result.answer
        assert result.metadata['task_type'] == 'algorithm'
        assert 'sort' in result.answer.lower()
    
    @pytest.mark.asyncio
    async def test_verify_valid_code(self, worker):
        """Test verification of valid code."""
        query = "Write a Python function"
        answer = """```python
def add(a, b):
    return a + b
```"""
        
        is_valid = await worker.verify(query, answer)
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_verify_invalid_code(self, worker):
        """Test verification fails for invalid code."""
        query = "Write a Python function"
        answer = """```python
def add(a, b)  # Missing colon
    return a + b
```"""
        
        is_valid = await worker.verify(query, answer)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_verify_no_code(self, worker):
        """Test verification fails when no code present."""
        query = "Write a function"
        answer = "I cannot write code for this."
        
        is_valid = await worker.verify(query, answer)
        assert is_valid is False


# ==================== Performance Tests ====================

class TestCodeWorkerPerformance:
    """Test CodeWorker performance characteristics."""
    
    @pytest.fixture
    def worker(self):
        """Create CodeWorker with mock LLM."""
        config = MockConfig()
        worker = CodeWorker.__new__(CodeWorker)
        worker.config = config
        worker.llm_client = MockLLMClient()
        worker.supported_languages = {'python', 'javascript'}
        return worker
    
    @pytest.mark.asyncio
    async def test_execution_time_reasonable(self, worker):
        """Test execution completes in reasonable time."""
        query = "Write a Python function to add numbers"
        result = await worker.solve(query)
        
        assert result.execution_time < 1.0  # Should be fast with mock
    
    def test_confidence_calculation(self, worker):
        """Test confidence score calculation."""
        # Valid Python code with syntax check
        confidence = worker._calculate_confidence(
            code="def add(a, b): return a + b",
            syntax_valid=True,
            task_type='generation',
            language='python'
        )
        assert confidence > 0.7
        
        # No code extracted
        confidence = worker._calculate_confidence(
            code=None,
            syntax_valid=False,
            task_type='generation',
            language='python'
        )
        assert confidence < 0.7
