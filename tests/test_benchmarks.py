"""Tests for the benchmark system."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path

from kaelum.benchmarks.dataset import (
    BenchmarkDataset,
    BenchmarkQuery,
    QueryCategory,
    DifficultyLevel,
    create_default_dataset
)
from kaelum.benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkResult,
    RunMode
)
from kaelum.benchmarks.evaluator import (
    BenchmarkEvaluator,
    EvaluationMetrics
)
from kaelum.core.workers import WorkerSpecialty
from kaelum.core.meta_reasoner import CombinationStrategy


class MockLLMClient:
    """Mock LLM client for testing."""
    
    async def generate(self, prompt: str) -> str:
        """Generate mock response."""
        if "math" in prompt.lower() or "solve" in prompt.lower():
            return "The answer is 42"
        elif "logic" in prompt.lower() or "prove" in prompt.lower():
            return "This follows from modus ponens"
        else:
            return "Mock response"


class MockConfig:
    """Mock config for workers."""
    def __init__(self):
        self.reasoning_llm = {"model": "mock-model"}


def mock_create_worker(specialty, config):
    """Mock worker factory that doesn't need real config."""
    from kaelum.core.workers import WorkerAgent, WorkerResult, WorkerSpecialty
    from kaelum.core.config import KaelumConfig
    
    class MockWorker(WorkerAgent):
        """Mock worker for testing."""
        
        def __init__(self, specialty):
            self.config = MockConfig()
            self._specialty = specialty
        
        def get_specialty(self) -> WorkerSpecialty:
            return self._specialty
        
        def can_handle(self, query: str, context=None) -> float:
            return 0.8
        
        async def solve(self, query: str, context=None) -> WorkerResult:
            """Required abstract method."""
            return await self.process(query, context)
        
        async def solve_async(self, query: str, context=None) -> WorkerResult:
            """Async solve for meta-reasoner."""
            return await self.process(query, context)
        
        async def process(self, query: str, context=None) -> WorkerResult:
            return WorkerResult(
                answer="42" if "math" in query.lower() else "mock answer",
                confidence=0.9,
                reasoning_steps=["Mock reasoning step 1", "Mock reasoning step 2"],
                verification_passed=True,
                specialty=self._specialty,
                execution_time=0.001,
                metadata={}
            )
        
        async def verify(self, query: str, answer: str, context=None) -> bool:
            return True
    
    return MockWorker(specialty)


# ==================== Dataset Tests ====================

class TestBenchmarkQuery:
    """Tests for BenchmarkQuery."""
    
    def test_create_query(self):
        """Test creating a query."""
        query = BenchmarkQuery(
            id="test_001",
            query="What is 2+2?",
            category=QueryCategory.MATH,
            difficulty=DifficultyLevel.EASY,
            expected_answer="4"
        )
        
        assert query.id == "test_001"
        assert query.query == "What is 2+2?"
        assert query.category == QueryCategory.MATH
        assert query.difficulty == DifficultyLevel.EASY
        assert query.expected_answer == "4"
    
    def test_query_to_dict(self):
        """Test converting query to dictionary."""
        query = BenchmarkQuery(
            id="test_001",
            query="Test query",
            category=QueryCategory.LOGIC,
            difficulty=DifficultyLevel.MEDIUM
        )
        
        data = query.to_dict()
        assert data["id"] == "test_001"
        assert data["category"] == "logic"
        assert data["difficulty"] == "medium"
    
    def test_query_from_dict(self):
        """Test creating query from dictionary."""
        data = {
            "id": "test_001",
            "query": "Test query",
            "category": "math",
            "difficulty": "hard",
            "expected_answer": "42"
        }
        
        query = BenchmarkQuery.from_dict(data)
        assert query.id == "test_001"
        assert query.category == QueryCategory.MATH
        assert query.difficulty == DifficultyLevel.HARD
        assert query.expected_answer == "42"


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset."""
    
    def test_create_dataset(self):
        """Test creating empty dataset."""
        dataset = BenchmarkDataset(name="test")
        assert dataset.name == "test"
        assert len(dataset) == 0
    
    def test_add_query(self):
        """Test adding queries."""
        dataset = BenchmarkDataset()
        query = BenchmarkQuery("q1", "Test?", QueryCategory.MATH, DifficultyLevel.EASY)
        
        dataset.add_query(query)
        assert len(dataset) == 1
    
    def test_add_multiple_queries(self):
        """Test adding multiple queries at once."""
        dataset = BenchmarkDataset()
        queries = [
            BenchmarkQuery("q1", "Test 1?", QueryCategory.MATH, DifficultyLevel.EASY),
            BenchmarkQuery("q2", "Test 2?", QueryCategory.LOGIC, DifficultyLevel.MEDIUM),
        ]
        
        dataset.add_queries(queries)
        assert len(dataset) == 2
    
    def test_filter_by_category(self):
        """Test filtering by category."""
        dataset = BenchmarkDataset()
        dataset.add_queries([
            BenchmarkQuery("q1", "Math?", QueryCategory.MATH, DifficultyLevel.EASY),
            BenchmarkQuery("q2", "Logic?", QueryCategory.LOGIC, DifficultyLevel.EASY),
            BenchmarkQuery("q3", "More math?", QueryCategory.MATH, DifficultyLevel.MEDIUM),
        ])
        
        math_queries = dataset.filter_by_category(QueryCategory.MATH)
        assert len(math_queries) == 2
    
    def test_filter_by_difficulty(self):
        """Test filtering by difficulty."""
        dataset = BenchmarkDataset()
        dataset.add_queries([
            BenchmarkQuery("q1", "Easy?", QueryCategory.MATH, DifficultyLevel.EASY),
            BenchmarkQuery("q2", "Hard?", QueryCategory.MATH, DifficultyLevel.HARD),
            BenchmarkQuery("q3", "Also easy?", QueryCategory.LOGIC, DifficultyLevel.EASY),
        ])
        
        easy_queries = dataset.filter_by_difficulty(DifficultyLevel.EASY)
        assert len(easy_queries) == 2
    
    def test_get_by_id(self):
        """Test getting query by ID."""
        dataset = BenchmarkDataset()
        query = BenchmarkQuery("q1", "Test?", QueryCategory.MATH, DifficultyLevel.EASY)
        dataset.add_query(query)
        
        found = dataset.get_by_id("q1")
        assert found is not None
        assert found.id == "q1"
        
        not_found = dataset.get_by_id("q999")
        assert not_found is None
    
    def test_save_and_load(self):
        """Test saving and loading dataset."""
        dataset = BenchmarkDataset(name="test_dataset")
        dataset.add_queries([
            BenchmarkQuery("q1", "Test 1?", QueryCategory.MATH, DifficultyLevel.EASY, "42"),
            BenchmarkQuery("q2", "Test 2?", QueryCategory.LOGIC, DifficultyLevel.MEDIUM),
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Save
            dataset.save(filepath)
            
            # Load
            loaded = BenchmarkDataset.load(filepath)
            assert loaded.name == "test_dataset"
            assert len(loaded) == 2
            assert loaded.get_by_id("q1").expected_answer == "42"
        finally:
            Path(filepath).unlink()
    
    def test_iteration(self):
        """Test iterating over dataset."""
        dataset = BenchmarkDataset()
        dataset.add_queries([
            BenchmarkQuery("q1", "Test 1?", QueryCategory.MATH, DifficultyLevel.EASY),
            BenchmarkQuery("q2", "Test 2?", QueryCategory.LOGIC, DifficultyLevel.MEDIUM),
        ])
        
        ids = [q.id for q in dataset]
        assert ids == ["q1", "q2"]


class TestDefaultDataset:
    """Tests for the default dataset."""
    
    def test_create_default_dataset(self):
        """Test creating default dataset."""
        dataset = create_default_dataset()
        
        assert dataset.name == "kaelum_v1"
        assert len(dataset) == 100  # Should have 100 queries
    
    def test_default_has_all_categories(self):
        """Test default dataset has all categories."""
        dataset = create_default_dataset()
        
        categories = {q.category for q in dataset}
        assert QueryCategory.MATH in categories
        assert QueryCategory.LOGIC in categories
        assert QueryCategory.CODE in categories
        assert QueryCategory.FACTUAL in categories
        assert QueryCategory.CREATIVE in categories
    
    def test_default_has_all_difficulties(self):
        """Test default dataset has all difficulties."""
        dataset = create_default_dataset()
        
        difficulties = {q.difficulty for q in dataset}
        assert DifficultyLevel.EASY in difficulties
        assert DifficultyLevel.MEDIUM in difficulties
        assert DifficultyLevel.HARD in difficulties


# ==================== Runner Tests ====================

class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""
    
    @pytest.fixture
    def llm_client(self):
        """Create mock LLM client."""
        return MockLLMClient()
    
    @pytest.fixture
    def runner(self, llm_client, monkeypatch):
        """Create benchmark runner."""
        # Patch create_worker to use our mock
        import kaelum.benchmarks.runner
        monkeypatch.setattr(kaelum.benchmarks.runner, 'create_worker', mock_create_worker)
        return BenchmarkRunner(llm_client)
    
    @pytest.fixture
    def test_query(self):
        """Create test query."""
        return BenchmarkQuery(
            "test_001",
            "What is 2+2?",
            QueryCategory.MATH,
            DifficultyLevel.EASY,
            "4"
        )
    
    @pytest.mark.asyncio
    async def test_run_single_worker(self, runner, test_query):
        """Test running with single worker."""
        result = await runner.run_single_worker(test_query, WorkerSpecialty.MATH)
        
        assert result.query_id == "test_001"
        assert result.mode == RunMode.SINGLE_WORKER
        assert result.confidence > 0
        assert len(result.workers_used) == 1
    
    @pytest.mark.asyncio
    async def test_run_meta_reasoner(self, runner, test_query):
        """Test running with meta-reasoner."""
        result = await runner.run_meta_reasoner(
            test_query,
            CombinationStrategy.WEIGHTED
        )
        
        assert result.query_id == "test_001"
        assert result.mode == RunMode.META_REASONER
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_run_comparison(self, runner, test_query):
        """Test running both modes."""
        results = await runner.run_comparison(
            test_query,
            WorkerSpecialty.MATH,
            CombinationStrategy.WEIGHTED
        )
        
        assert "single" in results
        assert "meta" in results
        assert results["single"].mode == RunMode.SINGLE_WORKER
        assert results["meta"].mode == RunMode.META_REASONER
    
    @pytest.mark.asyncio
    async def test_run_dataset(self, runner):
        """Test running multiple queries."""
        dataset = BenchmarkDataset()
        dataset.add_queries([
            BenchmarkQuery("q1", "What is 2+2?", QueryCategory.MATH, DifficultyLevel.EASY, "4"),
            BenchmarkQuery("q2", "What is 3+3?", QueryCategory.MATH, DifficultyLevel.EASY, "6"),
        ])
        
        results = await runner.run_dataset(dataset, mode=RunMode.SINGLE_WORKER)
        
        assert len(results) == 2
        assert all(r.mode == RunMode.SINGLE_WORKER for r in results)
    
    @pytest.mark.asyncio
    async def test_run_dataset_with_limit(self, runner):
        """Test running limited number of queries."""
        dataset = create_default_dataset()
        
        results = await runner.run_dataset(dataset, max_queries=5)
        
        assert len(results) == 5
    
    def test_get_specialty_for_category(self, runner):
        """Test category to specialty mapping."""
        assert runner._get_specialty_for_category("math") == WorkerSpecialty.MATH
        assert runner._get_specialty_for_category("logic") == WorkerSpecialty.LOGIC
        assert runner._get_specialty_for_category("code") == WorkerSpecialty.CODE
    
    def test_get_results_by_mode(self, runner):
        """Test filtering results by mode."""
        runner.results = [
            BenchmarkResult("q1", "Query 1", "Answer", 0.1, 0.9, True, RunMode.SINGLE_WORKER),
            BenchmarkResult("q2", "Query 2", "Answer", 0.1, 0.9, True, RunMode.META_REASONER),
            BenchmarkResult("q3", "Query 3", "Answer", 0.1, 0.9, True, RunMode.SINGLE_WORKER),
        ]
        
        single_results = runner.get_results_by_mode(RunMode.SINGLE_WORKER)
        assert len(single_results) == 2
    
    def test_summary_stats(self, runner):
        """Test getting summary statistics."""
        runner.results = [
            BenchmarkResult("q1", "Query 1", "Answer", 0.1, 0.9, True, RunMode.SINGLE_WORKER),
            BenchmarkResult("q2", "Query 2", "Answer", 0.2, 0.8, False, RunMode.META_REASONER),
        ]
        
        stats = runner.get_summary_stats()
        assert stats["total_queries"] == 2
        assert abs(stats["avg_execution_time"] - 0.15) < 0.01  # Allow floating point tolerance
        assert stats["verified_count"] == 1


# ==================== Evaluator Tests ====================

class TestBenchmarkEvaluator:
    """Tests for BenchmarkEvaluator."""
    
    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        dataset = BenchmarkDataset()
        dataset.add_queries([
            BenchmarkQuery("q1", "What is 2+2?", QueryCategory.MATH, DifficultyLevel.EASY, "4"),
            BenchmarkQuery("q2", "What is 3+3?", QueryCategory.MATH, DifficultyLevel.EASY, "6"),
            BenchmarkQuery("q3", "Hard question?", QueryCategory.LOGIC, DifficultyLevel.HARD),
        ])
        return dataset
    
    @pytest.fixture
    def evaluator(self, dataset):
        """Create evaluator."""
        return BenchmarkEvaluator(dataset)
    
    def test_normalize_answer(self, evaluator):
        """Test answer normalization."""
        assert evaluator._normalize_answer("42") == "42"
        assert evaluator._normalize_answer("  42  ") == "42"
        assert evaluator._normalize_answer("4 2") == "42"
        assert evaluator._normalize_answer("4,200") == "4200"
    
    def test_evaluate_correct_result(self, evaluator, dataset):
        """Test evaluating correct result."""
        query = dataset.get_by_id("q1")
        result = BenchmarkResult("q1", "What is 2+2?", "The answer is 4", 0.1, 0.9, True, RunMode.SINGLE_WORKER)
        
        assert evaluator.evaluate_result(result, query) is True
    
    def test_evaluate_incorrect_result(self, evaluator, dataset):
        """Test evaluating incorrect result."""
        query = dataset.get_by_id("q1")
        result = BenchmarkResult("q1", "What is 2+2?", "The answer is 5", 0.1, 0.9, True, RunMode.SINGLE_WORKER)
        
        assert evaluator.evaluate_result(result, query) is False
    
    def test_evaluate_no_expected_answer(self, evaluator, dataset):
        """Test evaluating when no expected answer."""
        query = dataset.get_by_id("q3")  # No expected answer
        result = BenchmarkResult("q3", "Hard?", "Answer", 0.1, 0.8, True, RunMode.SINGLE_WORKER)
        
        # Should use confidence threshold
        assert evaluator.evaluate_result(result, query) is True
        
        result_low_conf = BenchmarkResult("q3", "Hard?", "Answer", 0.1, 0.5, True, RunMode.SINGLE_WORKER)
        assert evaluator.evaluate_result(result_low_conf, query) is False
    
    def test_calculate_metrics(self, evaluator):
        """Test calculating metrics."""
        results = [
            BenchmarkResult("q1", "Query", "4", 0.1, 0.9, True, RunMode.SINGLE_WORKER),
            BenchmarkResult("q2", "Query", "6", 0.2, 0.8, False, RunMode.SINGLE_WORKER),
        ]
        
        metrics = evaluator.calculate_metrics(results)
        
        assert metrics.total_queries == 2
        assert metrics.accuracy == 1.0  # Both correct
        assert abs(metrics.avg_confidence - 0.85) < 0.01  # Allow floating point tolerance
        assert abs(metrics.avg_execution_time - 0.15) < 0.01  # Allow floating point tolerance
        assert metrics.verification_rate == 0.5
    
    def test_calculate_metrics_with_speedup(self, evaluator):
        """Test calculating metrics with speedup."""
        results = [
            BenchmarkResult("q1", "Query", "4", 0.1, 0.9, True, RunMode.META_REASONER),
        ]
        baseline = [
            BenchmarkResult("q1", "Query", "4", 0.2, 0.9, True, RunMode.SINGLE_WORKER),
        ]
        
        metrics = evaluator.calculate_metrics(results, baseline)
        assert metrics.speedup == 2.0  # 0.2 / 0.1
    
    def test_compare_modes(self, evaluator):
        """Test comparing single vs meta modes."""
        single_results = [
            BenchmarkResult("q1", "Q", "4", 0.2, 0.8, True, RunMode.SINGLE_WORKER),
            BenchmarkResult("q2", "Q", "6", 0.2, 0.7, False, RunMode.SINGLE_WORKER),
        ]
        meta_results = [
            BenchmarkResult("q1", "Q", "4", 0.1, 0.9, True, RunMode.META_REASONER),
            BenchmarkResult("q2", "Q", "6", 0.1, 0.9, True, RunMode.META_REASONER),
        ]
        
        comparison = evaluator.compare_modes(single_results, meta_results)
        
        assert "single_worker" in comparison
        assert "meta_reasoner" in comparison
        assert "improvements" in comparison
        assert "recommendation" in comparison
    
    def test_analyze_by_category(self, evaluator):
        """Test analyzing by category."""
        results = [
            BenchmarkResult("q1", "Q", "4", 0.1, 0.9, True, RunMode.SINGLE_WORKER, metadata={"category": "math"}),
            BenchmarkResult("q2", "Q", "6", 0.1, 0.8, True, RunMode.SINGLE_WORKER, metadata={"category": "math"}),
            BenchmarkResult("q3", "Q", "A", 0.1, 0.7, False, RunMode.SINGLE_WORKER, metadata={"category": "logic"}),
        ]
        
        by_category = evaluator.analyze_by_category(results)
        
        assert "math" in by_category
        assert "logic" in by_category
        assert by_category["math"].total_queries == 2
        assert by_category["logic"].total_queries == 1
    
    def test_generate_report(self, evaluator):
        """Test generating comprehensive report."""
        results = [
            BenchmarkResult("q1", "Q", "4", 0.1, 0.9, True, RunMode.SINGLE_WORKER, metadata={"category": "math", "difficulty": "easy"}),
            BenchmarkResult("q2", "Q", "6", 0.1, 0.8, False, RunMode.SINGLE_WORKER, metadata={"category": "logic", "difficulty": "medium"}),
        ]
        
        report = evaluator.generate_report(results)
        
        assert "summary" in report
        assert "by_category" in report
        assert "by_difficulty" in report
        assert report["summary"]["total_queries"] == 2
    
    def test_save_report(self, evaluator):
        """Test saving report to file."""
        results = [
            BenchmarkResult("q1", "Q", "4", 0.1, 0.9, True, RunMode.SINGLE_WORKER, metadata={"category": "math", "difficulty": "easy"}),
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            report = evaluator.generate_report(results, output_file=filepath)
            
            # Check file was created
            assert Path(filepath).exists()
            
            # Load and verify
            with open(filepath, 'r') as f:
                loaded = json.load(f)
            assert loaded["summary"]["total_queries"] == 1
        finally:
            Path(filepath).unlink()


# ==================== Integration Tests ====================

class TestBenchmarkIntegration:
    """Integration tests for full benchmark flow."""
    
    @pytest.fixture
    def llm_client(self):
        """Create mock LLM client."""
        return MockLLMClient()
    
    @pytest.mark.asyncio
    async def test_full_benchmark_flow(self, llm_client, monkeypatch):
        """Test complete benchmark workflow."""
        # Patch create_worker
        import kaelum.benchmarks.runner
        monkeypatch.setattr(kaelum.benchmarks.runner, 'create_worker', mock_create_worker)
        
        # Create dataset
        dataset = BenchmarkDataset(name="integration_test")
        dataset.add_queries([
            BenchmarkQuery("q1", "What is 2+2?", QueryCategory.MATH, DifficultyLevel.EASY, "4"),
            BenchmarkQuery("q2", "What is 3+3?", QueryCategory.MATH, DifficultyLevel.EASY, "6"),
        ])
        
        # Run benchmarks
        runner = BenchmarkRunner(llm_client)
        results = await runner.run_dataset(dataset, mode=RunMode.SINGLE_WORKER)
        
        # Evaluate
        evaluator = BenchmarkEvaluator(dataset)
        metrics = evaluator.calculate_metrics(results)
        
        assert metrics.total_queries == 2
        assert metrics.accuracy >= 0.0
        assert metrics.avg_confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_comparison_workflow(self, llm_client, monkeypatch):
        """Test comparison workflow."""
        # Patch create_worker
        import kaelum.benchmarks.runner
        monkeypatch.setattr(kaelum.benchmarks.runner, 'create_worker', mock_create_worker)
        
        dataset = BenchmarkDataset(name="comparison_test")
        dataset.add_query(
            BenchmarkQuery("q1", "What is 2+2?", QueryCategory.MATH, DifficultyLevel.EASY, "4")
        )
        
        # Run both modes
        runner = BenchmarkRunner(llm_client)
        
        single_results = await runner.run_dataset(dataset, mode=RunMode.SINGLE_WORKER)
        runner.clear_results()
        meta_results = await runner.run_dataset(dataset, mode=RunMode.META_REASONER)
        
        # Compare
        evaluator = BenchmarkEvaluator(dataset)
        comparison = evaluator.compare_modes(single_results, meta_results)
        
        assert "single_worker" in comparison
        assert "meta_reasoner" in comparison
        assert "improvements" in comparison
