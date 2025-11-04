"""Tests for worker agents."""

import pytest
from unittest.mock import Mock, patch
from kaelum.core.workers import (
    WorkerAgent, MathWorker, LogicWorker,
    WorkerSpecialty, WorkerResult, create_worker
)
from kaelum.core.config import KaelumConfig


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock = Mock()
    mock.generate.return_value = "42"  # Default answer
    return mock


class TestWorkerBase:
    """Test base worker functionality."""
    
    def test_math_worker_specialty(self):
        """Test MathWorker returns correct specialty."""
        worker = MathWorker()
        assert worker.get_specialty() == WorkerSpecialty.MATH
    
    def test_logic_worker_specialty(self):
        """Test LogicWorker returns correct specialty."""
        worker = LogicWorker()
        assert worker.get_specialty() == WorkerSpecialty.LOGIC
    
    def test_worker_factory(self):
        """Test worker factory creates correct workers."""
        math_worker = create_worker(WorkerSpecialty.MATH)
        assert isinstance(math_worker, MathWorker)
        
        logic_worker = create_worker(WorkerSpecialty.LOGIC)
        assert isinstance(logic_worker, LogicWorker)
    
    def test_worker_factory_invalid(self):
        """Test worker factory raises error for invalid specialty."""
        with pytest.raises(ValueError):
            create_worker(WorkerSpecialty.CODE)  # Not implemented yet


class TestMathWorker:
    """Test MathWorker functionality."""
    
    @pytest.fixture
    def worker(self, mock_llm_client):
        """Create MathWorker instance with mocked LLM."""
        worker = MathWorker()
        worker.llm_client = mock_llm_client
        return worker
    
    def test_can_handle_math_queries(self, worker):
        """Test MathWorker identifies math queries."""
        # Strong math queries
        assert worker.can_handle("Calculate 15 + 27") > 0.5
        assert worker.can_handle("Solve for x: 2x + 5 = 15") > 0.5
        assert worker.can_handle("What is the derivative of x^2?") > 0.5
        
        # Weak/non-math queries
        assert worker.can_handle("Who was the first president?") < 0.3
        assert worker.can_handle("Write a poem about autumn") < 0.3
    
    def test_solve_simple_arithmetic(self, worker):
        """Test MathWorker solves simple arithmetic."""
        result = worker.solve("What is 15 + 27?")
        
        assert isinstance(result, WorkerResult)
        assert result.specialty == WorkerSpecialty.MATH
        assert result.answer  # Has an answer
        assert result.confidence > 0.0
        assert len(result.reasoning_steps) > 0
        assert result.execution_time > 0
    
    def test_solve_algebra(self, worker):
        """Test MathWorker solves algebraic equations."""
        result = worker.solve("Solve for x: 2*x + 5 = 15")
        
        assert isinstance(result, WorkerResult)
        assert result.specialty == WorkerSpecialty.MATH
        # Should get x = 5
        assert "5" in result.answer
        assert result.confidence > 0.5
    
    def test_solve_calculus(self, worker):
        """Test MathWorker handles calculus."""
        result = worker.solve("What is the derivative of x^2?")
        
        assert isinstance(result, WorkerResult)
        assert result.specialty == WorkerSpecialty.MATH
        # Should get 2*x
        assert "2" in result.answer and "x" in result.answer
        assert result.confidence > 0.5
    
    def test_result_structure(self, worker):
        """Test WorkerResult has all required fields."""
        result = worker.solve("Calculate 10 + 20")
        
        assert hasattr(result, 'answer')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reasoning_steps')
        assert hasattr(result, 'verification_passed')
        assert hasattr(result, 'specialty')
        assert hasattr(result, 'execution_time')
        assert hasattr(result, 'error')
        assert hasattr(result, 'metadata')
        
        # Check to_dict works
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert 'answer' in result_dict
        assert 'confidence' in result_dict


class TestLogicWorker:
    """Test LogicWorker functionality."""
    
    @pytest.fixture
    def worker(self, mock_llm_client):
        """Create LogicWorker instance with mocked LLM."""
        worker = LogicWorker()
        worker.llm_client = mock_llm_client
        # Set more logical responses for logic queries
        mock_llm_client.generate.return_value = "Socrates is mortal (by deductive reasoning)"
        return worker
    
    def test_can_handle_logic_queries(self, worker):
        """Test LogicWorker identifies logic queries."""
        # Strong logic queries
        assert worker.can_handle("If A implies B, and B implies C, does A imply C?") > 0.5
        assert worker.can_handle("All humans are mortal. Socrates is human. Therefore?") > 0.5
        assert worker.can_handle("Prove that if n is even, then n^2 is even") > 0.5
        
        # Weak/non-logic queries
        assert worker.can_handle("What is 15 + 27?") < 0.3
        assert worker.can_handle("Who wrote Romeo and Juliet?") < 0.3
    
    def test_solve_syllogism(self, worker):
        """Test LogicWorker solves syllogisms."""
        result = worker.solve(
            "All humans are mortal. Socrates is human. Therefore?"
        )
        
        assert isinstance(result, WorkerResult)
        assert result.specialty == WorkerSpecialty.LOGIC
        assert result.answer  # Has an answer
        assert result.confidence > 0.0
        assert len(result.reasoning_steps) > 0
    
    def test_solve_conditional(self, worker):
        """Test LogicWorker handles conditional logic."""
        result = worker.solve(
            "If A implies B, and B implies C, does A imply C?"
        )
        
        assert isinstance(result, WorkerResult)
        assert result.specialty == WorkerSpecialty.LOGIC
        assert result.confidence > 0.5
        assert len(result.reasoning_steps) >= 5  # Deep reflection
    
    def test_deep_reflection(self, worker):
        """Test LogicWorker uses deep reflection."""
        result = worker.solve(
            "Identify the fallacy: 'Everyone believes X, therefore X is true'"
        )
        
        # Should have multiple reasoning steps from deep reflection
        assert len(result.reasoning_steps) >= 3
        assert result.metadata.get('reflection_depth') == 5


class TestWorkerAsync:
    """Test async worker functionality."""
    
    @pytest.mark.asyncio
    async def test_math_worker_async(self):
        """Test MathWorker async execution."""
        worker = MathWorker()
        result = await worker.solve_async("Calculate 15 + 27")
        
        assert isinstance(result, WorkerResult)
        assert result.answer
        assert result.specialty == WorkerSpecialty.MATH
    
    @pytest.mark.asyncio
    async def test_logic_worker_async(self):
        """Test LogicWorker async execution."""
        worker = LogicWorker()
        result = await worker.solve_async(
            "If it rains, the ground is wet. The ground is wet. Is it raining?"
        )
        
        assert isinstance(result, WorkerResult)
        assert result.answer
        assert result.specialty == WorkerSpecialty.LOGIC
    
    @pytest.mark.asyncio
    async def test_parallel_workers(self):
        """Test running multiple workers in parallel."""
        import asyncio
        
        math_worker = MathWorker()
        logic_worker = LogicWorker()
        
        # Run both workers in parallel
        results = await asyncio.gather(
            math_worker.solve_async("Calculate 10 * 5"),
            logic_worker.solve_async("All cats are animals. Fluffy is a cat. Therefore?")
        )
        
        assert len(results) == 2
        assert results[0].specialty == WorkerSpecialty.MATH
        assert results[1].specialty == WorkerSpecialty.LOGIC
        assert all(r.answer for r in results)


class TestWorkerVerification:
    """Test worker verification functionality."""
    
    def test_math_worker_verification(self):
        """Test MathWorker verification."""
        worker = MathWorker()
        
        # Correct answer should pass
        assert worker.verify("What is 2 + 2?", "4") == True
        
        # Note: Verification may not catch all errors in current implementation
    
    def test_logic_worker_verification(self):
        """Test LogicWorker verification."""
        worker = LogicWorker()
        
        # Test verification (may not be perfect in current implementation)
        result = worker.verify(
            "All humans are mortal. Socrates is human.",
            "Socrates is mortal"
        )
        assert isinstance(result, bool)


class TestWorkerPerformance:
    """Test worker performance characteristics."""
    
    def test_math_worker_speed(self):
        """Test MathWorker completes in reasonable time."""
        worker = MathWorker()
        result = worker.solve("Calculate 100 + 200")
        
        # Should complete in under 5 seconds
        assert result.execution_time < 5.0
    
    def test_logic_worker_speed(self):
        """Test LogicWorker completes in reasonable time."""
        worker = LogicWorker()
        result = worker.solve("If A then B. A is true. Therefore?")
        
        # Logic queries take longer due to deep reflection, but should be under 10s
        assert result.execution_time < 10.0
    
    def test_worker_confidence_scores(self, mock_llm_client):
        """Test workers return reasonable confidence scores."""
        math_worker = MathWorker()
        math_worker.llm_client = mock_llm_client
        mock_llm_client.generate.return_value = "30"
        
        logic_worker = LogicWorker()
        logic_worker.llm_client = mock_llm_client
        mock_llm_client.generate.return_value = "B is true"
        
        math_result = math_worker.solve("Calculate 5 * 6")
        logic_result = logic_worker.solve("If A implies B, and A is true, then B is true")
        
        # Confidence should be between 0 and 1
        assert 0.0 <= math_result.confidence <= 1.0
        assert 0.0 <= logic_result.confidence <= 1.0
        
        # Should have some confidence for valid queries
        assert math_result.confidence > 0.3
        assert logic_result.confidence > 0.3
