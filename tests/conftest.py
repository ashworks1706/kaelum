"""Pytest configuration and fixtures."""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Set dummy API keys for testing if not already set
    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = "test-key-not-real"


@pytest.fixture
def sample_query():
    """Provide a sample query for testing."""
    return "If 3x + 5 = 11, what is x?"


@pytest.fixture
def sample_trace():
    """Provide a sample reasoning trace for testing."""
    return [
        "Start with the equation 3x + 5 = 11",
        "Subtract 5 from both sides: 3x = 6",
        "Divide both sides by 3: x = 2",
    ]


@pytest.fixture
def sample_context():
    """Provide sample context for testing."""
    return "This is a basic algebra problem."
