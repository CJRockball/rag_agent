"""Configuration and fixtures for the test suite."""

import pytest
import os
from unittest.mock import Mock, patch
from langchain_core.documents import Document


@pytest.fixture(scope="session")
def test_environment():
    """Set up test environment variables."""
    test_env = {
        "GOOGLE_API_KEY": "test_api_key_for_testing",
        "CHROMA_DB_PATH": "test_db_path",
        "COLLECTION_NAME": "test_collection",
    }

    with patch.dict(os.environ, test_env):
        yield test_env


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        Document(
            page_content="Artificial Intelligence (AI) is the simulation \
                of human intelligence in machines.",
            metadata={
                "source": "ai_basics.pdf",
                "page": 1,
            },
        ),
        Document(
            page_content="Machine Learning is a subset of AI that enables \
                computers to learn without being explicitly programmed.",
            metadata={"source": "ml_intro.pdf", "page": 1},
        ),
        Document(
            page_content="Deep Learning uses neural networks with \
                multiple layers to model and understand complex patterns.",
            metadata={"source": "dl_guide.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def sample_state():
    """Provide sample state for testing."""
    return {
        "question": "What is artificial intelligence?",
        "context": [
            Document(
                page_content="AI is the simulation of human intelligence \
                    in machines."
            ),
            Document(
                page_content="Machine learning is a subset of \
                    artificial intelligence."
            ),
            Document(page_content="Deep learning uses neural networks."),
        ],
        "answer": "AI is the simulation of human intelligence in machines.",
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    mock = Mock()
    mock.content = "This is a mock response from the language model."
    return mock


@pytest.fixture
def mock_chroma_db():
    """Mock ChromaDB for testing."""
    mock_db = Mock()
    mock_db.similarity_search.return_value = [
        Document(page_content="Mock document content 1"),
        Document(page_content="Mock document content 2"),
        Document(page_content="Mock document content 3"),
    ]
    return mock_db


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    mock = Mock()
    mock.embed_query.return_value = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    return mock


@pytest.fixture(autouse=True)
def reset_streamlit_state():
    """Reset Streamlit session state before each test."""
    # This fixture automatically runs before each test
    # to ensure clean state for Streamlit tests
    import streamlit as st

    if hasattr(st, "session_state"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]


class MockStreamlitApp:
    """Mock Streamlit app for testing."""

    def __init__(self):
        self.title = []
        self.markdown = []
        self.chat_input = []
        self.chat_message = []
        self.exception = None
        self.session_state = {}

    def run(self):
        """Simulate running the app."""
        pass

    def set_value(self, value):
        """Set input value."""
        self.value = value
        return self


@pytest.fixture
def mock_streamlit_app():
    """Provide mock Streamlit app for testing."""
    return MockStreamlitApp()


# Test data fixtures
@pytest.fixture
def test_questions():
    """Provide test questions."""
    return [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the applications of deep learning?",
        "What is the difference between AI and ML?",
        "How do neural networks process information?",
    ]


@pytest.fixture
def test_responses():
    """Provide test responses."""
    return [
        "AI is the simulation of human intelligence in machines.",
        "Machine learning uses algorithms to find patterns in data.",
        "Deep learning is used in image recognition, NLP, and more.",
        "AI is the broader concept, while ML is a subset of AI.",
        "Neural networks process information through interconnected nodes.",
    ]


# Performance testing fixtures
@pytest.fixture
def performance_monitor():
    """Monitor performance during tests."""
    import time
    import psutil

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None

        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss

        def stop(self):
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            return {
                "duration": end_time - self.start_time,
                "memory_used": end_memory - self.start_memory,
            }

    return PerformanceMonitor()


# Error simulation fixtures
@pytest.fixture
def network_error():
    """Simulate network errors."""
    return ConnectionError("Network connection failed")


@pytest.fixture
def api_error():
    """Simulate API errors."""
    return Exception("API service unavailable")


@pytest.fixture
def timeout_error():
    """Simulate timeout errors."""
    return TimeoutError("Request timed out")


# Database fixtures
@pytest.fixture
def empty_db_response():
    """Simulate empty database response."""
    return []


@pytest.fixture
def large_db_response():
    """Simulate large database response."""
    return [Document(page_content=f"Document {i} content") for i in range(100)]


# Configuration for test runs
def pytest_configure(config):
    """Configure pytest settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m 'not slow'')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests",
    )
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance tests",
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(
            marker.name in ["integration", "performance"]
            for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)

        # Add slow marker to performance tests
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.performance)

        # Add integration marker to integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
