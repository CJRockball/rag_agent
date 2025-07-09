"""Test suite for the RAG agent core functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Any
import os

# Import the functions we want to test
# Note: You'll need to adjust these imports based on your actual file structure
# from agent_core import ask_rag, retrieve, generate, State, graph


class TestRAGAgent:
    """Test suite for the RAG agent functionality."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state for testing."""
        return {
            "question": "What is artificial intelligence?",
            "context": [
                Document(
                    page_content="AI is the simulation of human intelligence in machines."
                ),
                Document(
                    page_content="Machine learning is a subset of artificial intelligence."
                ),
                Document(
                    page_content="Deep learning uses neural networks with multiple layers."
                ),
            ],
            "answer": "AI is the simulation of human intelligence in machines.",
        }

    @pytest.fixture
    def mock_documents(self):
        """Create mock documents for testing."""
        return [
            Document(
                page_content="AI is the simulation of human intelligence in machines."
            ),
            Document(
                page_content="Machine learning is a subset of artificial intelligence."
            ),
            Document(
                page_content="Deep learning uses neural networks with multiple layers."
            ),
        ]

    @patch("src.agent.agent_core.db")
    def test_retrieve_function(self, mock_db, mock_state):
        """Test the retrieve function returns correct context."""
        from src.agent.agent_core import retrieve

        # Mock the similarity search
        mock_db.similarity_search.return_value = mock_state["context"]

        # Test the retrieve function
        result = retrieve(mock_state)

        # Assertions
        assert "context" in result
        assert len(result["context"]) == 3
        assert all(isinstance(doc, Document) for doc in result["context"])
        mock_db.similarity_search.assert_called_once_with(
            query=mock_state["question"], k=3
        )

    @patch("src.agent.agent_core.llm")
    def test_generate_function(self, mock_llm, mock_state):
        """Test the generate function produces correct response."""
        from src.agent.agent_core import generate

        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = (
            "AI is the simulation of human intelligence in machines."
        )
        mock_llm.invoke.return_value = mock_response

        # Test the generate function
        result = generate(mock_state)

        # Assertions
        assert "answer" in result
        assert result["answer"] == mock_response
        mock_llm.invoke.assert_called_once()

        # Check that the invoke was called with proper message structure
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert isinstance(call_args[0], SystemMessage)
        assert isinstance(call_args[1], HumanMessage)

    @patch("src.agent.agent_core.graph")
    def test_ask_rag_function(self, mock_graph):
        """Test the ask_rag function end-to-end."""
        from src.agent.agent_core import ask_rag

        # Mock the graph stream
        mock_response = Mock()
        mock_response.content = (
            "AI is the simulation of human intelligence in machines."
        )

        mock_graph.stream.return_value = [
            {"question": "What is AI?"},
            {"context": []},
            {"answer": mock_response},
        ]

        # Test the ask_rag function
        result = ask_rag("What is AI?")

        # Assertions
        assert result == "AI is the simulation of human intelligence in machines."
        mock_graph.stream.assert_called_once_with(
            {"question": "What is AI?"}, stream_mode="values"
        )

    @patch("src.agent.agent_core.graph")
    def test_ask_rag_no_answer(self, mock_graph):
        """Test ask_rag when no answer is found."""
        from src.agent.agent_core import ask_rag

        # Mock the graph stream with no answer
        mock_graph.stream.return_value = [{"question": "What is AI?"}, {"context": []}]

        # Test the ask_rag function
        result = ask_rag("What is AI?")

        # Assertions
        assert result is None

    @patch("src.agent.agent_core.graph")
    def test_ask_rag_with_string_answer(self, mock_graph):
        """Test ask_rag when answer is a string instead of object with content."""
        from src.agent.agent_core import ask_rag

        # Mock the graph stream with string answer
        mock_graph.stream.return_value = [
            {"question": "What is AI?"},
            {"context": []},
            {"answer": "Direct string answer"},
        ]

        # Test the ask_rag function
        result = ask_rag("What is AI?")

        # Assertions
        assert result == "Direct string answer"

    def test_state_structure(self):
        """Test that the State TypedDict has correct structure."""
        from src.agent.agent_core import State

        # Test creating a state instance
        state = State(
            question="Test question",
            context=[Document(page_content="Test content")],
            answer="Test answer",
        )

        assert "question" in state
        assert "context" in state
        assert "answer" in state
        assert isinstance(state["context"], list)
        assert isinstance(state["context"][0], Document)


class TestRAGAgentIntegration:
    """Integration tests for the RAG agent."""

    @pytest.fixture
    def setup_test_environment(self):
        """Set up test environment with proper configurations."""
        # Mock environment variables
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key"}):
            yield

    @patch("src.agent.agent_core.Chroma")
    @patch("src.agent.agent_core.ChatGoogleGenerativeAI")
    @patch("src.agent.agent_core.GoogleGenerativeAIEmbeddings")
    def test_component_initialization(
        self, mock_embeddings, mock_llm, mock_chroma, setup_test_environment
    ):
        """Test that all components are properly initialized."""
        # Import after patching
        import src.agent.agent_core

        # Verify that components are initialized
        assert src.agent.agent_core.llm is not None
        assert src.agent.agent_core.doc_embeddings is not None
        assert src.agent.agent_core.db is not None
        assert src.agent.agent_core.graph is not None

    @patch("src.agent.agent_core.db")
    @patch("src.agent.agent_core.llm")
    def test_full_rag_pipeline(self, mock_llm, mock_db):
        """Test the full RAG pipeline from question to answer."""
        from src.agent.agent_core import ask_rag

        # Mock the database response
        mock_documents = [
            Document(page_content="AI is the simulation of human intelligence."),
            Document(page_content="Machine learning is a subset of AI."),
            Document(page_content="Deep learning uses neural networks."),
        ]
        mock_db.similarity_search.return_value = mock_documents

        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = (
            "AI is the simulation of human intelligence in machines."
        )
        mock_llm.invoke.return_value = mock_response

        # Mock the graph to simulate the actual flow
        with patch("src.agent.agent_core.graph") as mock_graph:
            mock_graph.stream.return_value = [
                {"question": "What is AI?"},
                {"context": mock_documents},
                {"answer": mock_response},
            ]

            result = ask_rag("What is AI?")

            assert result == "AI is the simulation of human intelligence in machines."


class TestRAGAgentErrorHandling:
    """Test error handling in the RAG agent."""

    @patch("src.agent.agent_core.db")
    def test_retrieve_with_empty_results(self, mock_db):
        """Test retrieve function with empty search results."""
        from src.agent.agent_core import retrieve

        mock_db.similarity_search.return_value = []

        state = {"question": "What is AI?"}
        result = retrieve(state)

        assert result["context"] == []

    @patch("src.agent.agent_core.llm")
    def test_generate_with_llm_error(self, mock_llm):
        """Test generate function when LLM throws an error."""
        from src.agent.agent_core import generate

        mock_llm.invoke.side_effect = Exception("LLM service unavailable")

        state = {
            "question": "What is AI?",
            "context": [Document(page_content="Test content")],
        }

        with pytest.raises(Exception):
            generate(state)

    @patch("src.agent.agent_core.graph")
    def test_ask_rag_with_graph_error(self, mock_graph):
        """Test ask_rag when graph throws an error."""
        from src.agent.agent_core import ask_rag

        mock_graph.stream.side_effect = Exception("Graph execution failed")

        with pytest.raises(Exception):
            ask_rag("What is AI?")


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])
