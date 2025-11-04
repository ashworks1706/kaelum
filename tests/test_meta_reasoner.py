"""Tests for MetaReasoner system."""

import pytest
from unittest.mock import Mock
from kaelum.core.meta_reasoner import (
    MetaReasoner, MetaResult, CombinationStrategy
)
from kaelum.core.workers import (
    MathWorker, LogicWorker, WorkerResult, WorkerSpecialty
)


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock = Mock()
    mock.generate.return_value = "Synthesized answer: 42"
    return mock


@pytest.fixture
def mock_math_worker(mock_llm_client):
    """Create mocked MathWorker."""
    worker = MathWorker()
    worker.llm_client = mock_llm_client
    return worker


@pytest.fixture
def mock_logic_worker(mock_llm_client):
    """Create mocked LogicWorker."""
    worker = LogicWorker()
    worker.llm_client = mock_llm_client
    mock_llm_client.generate.return_value = "B is true"
    return worker


class TestMetaReasoner:
    """Test MetaReasoner basic functionality."""
    
    def test_add_worker(self, mock_math_worker):
        """Test adding single worker."""
        meta = MetaReasoner()
        meta.add_worker(mock_math_worker)
        
        assert len(meta.workers) == 1
        assert meta.workers[0].get_specialty() == WorkerSpecialty.MATH
    
    def test_add_workers(self, mock_math_worker, mock_logic_worker):
        """Test adding multiple workers."""
        meta = MetaReasoner()
        meta.add_workers([mock_math_worker, mock_logic_worker])
        
        assert len(meta.workers) == 2
        assert meta.workers[0].get_specialty() == WorkerSpecialty.MATH
        assert meta.workers[1].get_specialty() == WorkerSpecialty.LOGIC
    
    def test_reason_with_confidence_strategy(self, mock_math_worker, mock_logic_worker):
        """Test reasoning with confidence strategy."""
        meta = MetaReasoner()
        meta.add_workers([mock_math_worker, mock_logic_worker])
        
        # Math query should be routed to math worker
        result = meta.reason(
            "Calculate 15 + 27",
            strategy=CombinationStrategy.CONFIDENCE
        )
        
        assert isinstance(result, MetaResult)
        assert result.answer == "42"  # From SymPy
        assert result.confidence > 0.9
        assert result.strategy == CombinationStrategy.CONFIDENCE
        assert len(result.worker_results) > 0
        assert result.execution_time > 0
    
    def test_reason_with_voting_strategy(self, mock_math_worker, mock_logic_worker):
        """Test reasoning with voting strategy."""
        meta = MetaReasoner()
        meta.add_workers([mock_math_worker, mock_logic_worker])
        
        result = meta.reason(
            "What is 10 + 20?",
            strategy=CombinationStrategy.VOTING
        )
        
        assert isinstance(result, MetaResult)
        assert result.strategy == CombinationStrategy.VOTING
        assert "vote" in result.reasoning.lower()
        assert "votes" in result.metadata
    
    def test_reason_with_verification_strategy(self, mock_math_worker, mock_logic_worker):
        """Test reasoning with verification strategy."""
        meta = MetaReasoner()
        meta.add_workers([mock_math_worker, mock_logic_worker])
        
        result = meta.reason(
            "Calculate 5 * 6",
            strategy=CombinationStrategy.VERIFICATION
        )
        
        assert isinstance(result, MetaResult)
        assert result.strategy == CombinationStrategy.VERIFICATION
        assert "verified" in result.reasoning.lower()
    
    def test_reason_with_weighted_strategy(self, mock_math_worker, mock_logic_worker):
        """Test reasoning with weighted strategy."""
        meta = MetaReasoner()
        meta.add_workers([mock_math_worker, mock_logic_worker])
        
        result = meta.reason(
            "What is 2 + 2?",
            strategy=CombinationStrategy.WEIGHTED
        )
        
        assert isinstance(result, MetaResult)
        assert result.strategy == CombinationStrategy.WEIGHTED
        assert "weight" in result.reasoning.lower()
        assert "best_weight" in result.metadata


class TestMetaReasonerAsync:
    """Test MetaReasoner async functionality."""
    
    @pytest.mark.asyncio
    async def test_reason_async(self, mock_math_worker, mock_logic_worker):
        """Test async reasoning."""
        meta = MetaReasoner()
        meta.add_workers([mock_math_worker, mock_logic_worker])
        
        result = await meta.reason_async(
            "Calculate 100 + 200",
            strategy=CombinationStrategy.CONFIDENCE
        )
        
        assert isinstance(result, MetaResult)
        assert result.answer == "300"
        assert result.confidence > 0.9
    
    @pytest.mark.asyncio
    async def test_parallel_worker_execution(self, mock_math_worker, mock_logic_worker):
        """Test that workers execute in parallel."""
        meta = MetaReasoner()
        meta.add_workers([mock_math_worker, mock_logic_worker])
        
        # Query that both can handle
        result = await meta.reason_async(
            "Calculate 2 + 2 if true",
            strategy=CombinationStrategy.CONFIDENCE
        )
        
        assert isinstance(result, MetaResult)
        # Should have results from workers that could handle it
        assert len(result.worker_results) >= 1


class TestCombinationStrategies:
    """Test different combination strategies."""
    
    def test_voting_picks_majority(self):
        """Test voting strategy picks majority answer."""
        meta = MetaReasoner()
        
        # Create mock results with same answer
        results = [
            WorkerResult(
                answer="42",
                confidence=0.8,
                reasoning_steps=[],
                verification_passed=True,
                specialty=WorkerSpecialty.MATH,
                execution_time=0.01,
                error=None,
                metadata={}
            ),
            WorkerResult(
                answer="42",
                confidence=0.7,
                reasoning_steps=[],
                verification_passed=True,
                specialty=WorkerSpecialty.LOGIC,
                execution_time=0.01,
                error=None,
                metadata={}
            ),
            WorkerResult(
                answer="40",
                confidence=0.9,
                reasoning_steps=[],
                verification_passed=True,
                specialty=WorkerSpecialty.MATH,
                execution_time=0.01,
                error=None,
                metadata={}
            )
        ]
        
        combined = meta._combine_results(results, CombinationStrategy.VOTING, "test")
        
        assert combined["answer"] == "42"  # Majority
        assert combined["metadata"]["votes"] == 2
        assert "vote" in combined["reasoning"].lower()
    
    def test_confidence_picks_highest(self):
        """Test confidence strategy picks highest confidence."""
        meta = MetaReasoner()
        
        results = [
            WorkerResult(
                answer="42",
                confidence=0.7,
                reasoning_steps=[],
                verification_passed=True,
                specialty=WorkerSpecialty.MATH,
                execution_time=0.01,
                error=None,
                metadata={}
            ),
            WorkerResult(
                answer="43",
                confidence=0.95,
                reasoning_steps=[],
                verification_passed=True,
                specialty=WorkerSpecialty.LOGIC,
                execution_time=0.01,
                error=None,
                metadata={}
            )
        ]
        
        combined = meta._combine_results(results, CombinationStrategy.CONFIDENCE, "test")
        
        assert combined["answer"] == "43"  # Highest confidence
        assert combined["confidence"] == 0.95
    
    def test_verification_picks_verified(self):
        """Test verification strategy picks verified results."""
        meta = MetaReasoner()
        
        results = [
            WorkerResult(
                answer="42",
                confidence=0.9,
                reasoning_steps=[],
                verification_passed=False,
                specialty=WorkerSpecialty.MATH,
                execution_time=0.01,
                error=None,
                metadata={}
            ),
            WorkerResult(
                answer="43",
                confidence=0.7,
                reasoning_steps=[],
                verification_passed=True,
                specialty=WorkerSpecialty.LOGIC,
                execution_time=0.01,
                error=None,
                metadata={}
            )
        ]
        
        combined = meta._combine_results(results, CombinationStrategy.VERIFICATION, "test")
        
        assert combined["answer"] == "43"  # Verified
        assert "verified" in combined["reasoning"].lower()


class TestMetaReasonerFiltering:
    """Test worker filtering functionality."""
    
    def test_worker_filter_by_specialty(self, mock_math_worker, mock_logic_worker):
        """Test filtering workers by specialty."""
        meta = MetaReasoner()
        meta.add_workers([mock_math_worker, mock_logic_worker])
        
        # Filter to only math worker
        result = meta.reason(
            "Calculate 10 + 20",
            strategy=CombinationStrategy.CONFIDENCE,
            worker_filter=[WorkerSpecialty.MATH]
        )
        
        assert isinstance(result, MetaResult)
        # Should only have math worker result
        assert all(r.specialty == WorkerSpecialty.MATH for r in result.worker_results)
    
    def test_auto_worker_selection(self, mock_math_worker, mock_logic_worker):
        """Test automatic worker selection based on can_handle."""
        meta = MetaReasoner()
        meta.add_workers([mock_math_worker, mock_logic_worker])
        
        # Math query - should primarily use math worker
        result = meta.reason(
            "What is 15 + 27?",
            strategy=CombinationStrategy.CONFIDENCE
        )
        
        # Check that worker was selected
        assert len(result.worker_results) > 0
        assert result.metadata["num_workers"] > 0


class TestMetaReasonerErrorHandling:
    """Test error handling in MetaReasoner."""
    
    def test_handles_all_workers_failing(self):
        """Test graceful handling when all workers fail."""
        meta = MetaReasoner()
        
        # Create worker that will fail
        failing_worker = MathWorker()
        failing_worker.llm_client = None  # Will cause error
        meta.add_worker(failing_worker)
        
        result = meta.reason("test query")
        
        # Should return error result, not crash
        assert isinstance(result, MetaResult)
        assert "error" in result.metadata or result.confidence == 0.0
    
    def test_result_to_dict(self, mock_math_worker):
        """Test MetaResult serialization."""
        meta = MetaReasoner()
        meta.add_worker(mock_math_worker)
        
        result = meta.reason("Calculate 5 + 5")
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "answer" in result_dict
        assert "confidence" in result_dict
        assert "strategy" in result_dict
        assert "worker_results" in result_dict
        assert isinstance(result_dict["worker_results"], list)
