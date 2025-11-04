"""Tests for reflection engine."""

import pytest
from unittest.mock import Mock, MagicMock
from kaelum.core.reflection import ReflectionEngine


class TestReflectionEngine:
    """Test reflection engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        mock_client = Mock()
        engine = ReflectionEngine(mock_client, max_iterations=3)
        
        assert engine.llm == mock_client
        assert engine.max_iterations == 3
    
    def test_no_reflection_needed(self):
        """Test when reflection isn't needed (perfect trace)."""
        mock_client = Mock()
        mock_client.generate = Mock(return_value="No issues found")
        
        engine = ReflectionEngine(mock_client, max_iterations=2)
        
        query = "Test query"
        trace = ["Step 1: Correct", "Step 2: Correct"]
        
        # Mock _verify_trace to return no issues
        engine._verify_trace = Mock(return_value=[])
        
        result = engine.enhance_reasoning(query, trace)
        
        # Should return original trace if no issues
        assert len(result) > 0
    
    def test_reflection_improvement(self):
        """Test reflection improves trace."""
        mock_client = Mock()
        
        engine = ReflectionEngine(mock_client, max_iterations=2)
        
        query = "Solve 2x = 4"
        initial_trace = ["Step 1: x = 3"]  # Wrong answer
        
        # Mock verification to find issues first time, then none
        engine._verify_trace = Mock(side_effect=[
            ["Issue: x should be 2, not 3"],
            []  # Second verification passes
        ])
        
        # Mock improvement to return corrected trace
        engine._improve_trace = Mock(return_value=["Step 1: x = 2"])
        
        result = engine.enhance_reasoning(query, initial_trace)
        
        # Should have called improve_trace once
        assert engine._improve_trace.call_count == 1
    
    def test_max_iterations_respected(self):
        """Test that max iterations limit is respected."""
        mock_client = Mock()
        
        engine = ReflectionEngine(mock_client, max_iterations=2)
        
        query = "Test query"
        trace = ["Step 1"]
        
        # Always return issues (never fixes)
        engine._verify_trace = Mock(return_value=["Issue 1"])
        engine._improve_trace = Mock(return_value=trace)
        
        result = engine.enhance_reasoning(query, trace)
        
        # Should stop after max iterations
        assert engine._verify_trace.call_count <= 2


class TestReflectionMethods:
    """Test reflection helper methods."""
    
    def test_verify_trace_method(self):
        """Test _verify_trace method."""
        mock_client = Mock()
        mock_response = "Step 1 has a calculation error"
        mock_client.generate = Mock(return_value=mock_response)
        
        engine = ReflectionEngine(mock_client)
        
        issues = engine._verify_trace("query", ["Step 1: Wrong"])
        
        assert isinstance(issues, list)
        assert mock_client.generate.called
    
    def test_improve_trace_method(self):
        """Test _improve_trace method."""
        mock_client = Mock()
        mock_response = "1. Corrected step one\n2. Corrected step two"
        mock_client.generate = Mock(return_value=mock_response)
        
        engine = ReflectionEngine(mock_client)
        
        improved = engine._improve_trace(
            "query",
            ["Step 1: Wrong"],
            ["Issue: calculation error"]
        )
        
        assert isinstance(improved, list)
        assert len(improved) > 0
        assert mock_client.generate.called
