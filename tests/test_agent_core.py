"""Test suite for the RAG agent core functionality."""

import pytest
import os
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma


class TestRAGAgent:
    """Test suite for the RAG agent functionality."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state for testing."""
        mock_db = Mock(spec=Chroma)
        return {
            "question": "What is artificial intelligence?",
            "context": [
                Document(
                    page_content="AI is the simulation of \
                        human intelligence in machines."
                ),
                Document(
                    page_content="Machine learning is a subset of \
                        artificial intelligence."
                ),
                Document(
                    page_content="Deep learning uses \
                        neural networks with multiple layers."
                ),
            ],
            "answer": "AI is the simulation of human intelligence \
                in machines.",
            "db": mock_db,
        }

    @pytest.fixture
    def mock_documents(self):
        """Create mock documents for testing."""
        return [
            Document(
                page_content="AI is the simulation of \
                    human intelligence in machines."
            ),
            Document(
                page_content="Machine learning is a subset of \
                    artificial intelligence."
            ),
            Document(
                page_content="Deep learning uses \
                    neural networks with multiple layers."
            ),
        ]

    @pytest.fixture
    def mock_chroma_db(self):
        """Create a mock Chroma database."""
        mock_db = Mock(spec=Chroma)
        mock_db.similarity_search.return_value = [
            Document(page_content="Test document 1"),
            Document(page_content="Test document 2"),
            Document(page_content="Test document 3"),
        ]
        return mock_db

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def test_retrieve_function(self, mock_state, mock_documents):
        """Test the retrieve function returns correct context."""
        from src.agent.agent_core import retrieve

        # Mock the database in the state
        mock_state["db"].similarity_search.return_value = mock_documents

        # Test the retrieve function
        result = retrieve(mock_state)

        # Assertions
        assert "context" in result
        assert len(result["context"]) == 3
        assert all(isinstance(doc, Document) for doc in result["context"])
        mock_state["db"].similarity_search.assert_called_once_with(
            query=mock_state["question"], k=3
        )

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
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

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    @patch("src.agent.agent_core.graph")
    def test_ask_rag_with_connection_function(
        self, mock_graph, mock_chroma_db
    ):
        """Test the ask_rag_with_connection function end-to-end."""
        from src.agent.agent_core import ask_rag_with_connection

        # Mock the graph stream
        mock_response = Mock()
        mock_response.content = (
            "AI is the simulation of human intelligence in machines."
        )

        mock_graph.stream.return_value = [
            {"question": "What is AI?", "db": mock_chroma_db},
            {"context": []},
            {"answer": mock_response},
        ]

        # Test the ask_rag_with_connection function
        result = ask_rag_with_connection("What is AI?", mock_chroma_db)

        # Assertions
        assert (
            result
            == "AI is the simulation of \
                human intelligence in machines."
        )
        mock_graph.stream.assert_called_once_with(
            {"question": "What is AI?", "db": mock_chroma_db},
            stream_mode="values",
        )

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    @patch("src.agent.agent_core.graph")
    def test_ask_rag_no_answer(self, mock_graph, mock_chroma_db):
        """Test ask_rag_with_connection when no answer is found."""
        from src.agent.agent_core import ask_rag_with_connection

        # Mock the graph stream with no answer
        mock_graph.stream.return_value = [
            {"question": "What is AI?", "db": mock_chroma_db},
            {"context": []},
        ]

        # Test the ask_rag_with_connection function
        result = ask_rag_with_connection("What is AI?", mock_chroma_db)

        # Assertions
        assert result is None

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    @patch("src.agent.agent_core.graph")
    def test_ask_rag_with_string_answer(self, mock_graph, mock_chroma_db):
        """Test ask_rag_with_connection when answer is a string."""
        from src.agent.agent_core import ask_rag_with_connection

        # Mock the graph stream with string answer
        mock_graph.stream.return_value = [
            {"question": "What is AI?", "db": mock_chroma_db},
            {"context": []},
            {"answer": "Direct string answer"},
        ]

        # Test the ask_rag_with_connection function
        result = ask_rag_with_connection("What is AI?", mock_chroma_db)

        # Assertions
        assert result == "Direct string answer"

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def test_state_structure(self):
        """Test that the State TypedDict has correct structure."""
        from src.agent.agent_core import State
        from langchain_chroma import Chroma

        # Create a mock Chroma database
        mock_db = Mock(spec=Chroma)

        # Test creating a state instance
        state = State(
            question="Test question",
            context=[Document(page_content="Test content")],
            answer="Test answer",
            db=mock_db,
        )

        assert "question" in state
        assert "context" in state
        assert "answer" in state
        assert "db" in state
        assert isinstance(state["context"], list)
        assert isinstance(state["context"][0], Document)


class TestRAGAgentIntegration:
    """Integration tests for the RAG agent."""

    @pytest.fixture
    def setup_test_environment(self):
        """Set up test environment with proper configurations."""
        test_env = {
            "GOOGLE_API_KEY": "test_api_key",
            "CHROMA_DB_PATH": "test_db_path",
            "COLLECTION_NAME": "test_collection",
        }
        with patch.dict(os.environ, test_env):
            yield

    @patch("src.agent.agent_core.ChatGoogleGenerativeAI")
    def test_component_initialization(
        self, mock_llm_class, setup_test_environment
    ):
        """Test that all components are properly initialized."""
        # Import after environment setup
        import src.agent.agent_core

        # Verify that components are initialized
        assert src.agent.agent_core.llm is not None
        mock_llm_class.assert_called_once()

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    @patch("src.agent.agent_core.llm")
    def test_full_rag_pipeline(self, mock_llm, mock_chroma_db):
        """Test the full RAG pipeline from question to answer."""
        from src.agent.agent_core import ask_rag_with_connection

        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = (
            "AI is the simulation of human intelligence in machines."
        )
        mock_llm.invoke.return_value = mock_response

        # Mock the database response
        mock_documents = [
            Document(
                page_content="AI is the simulation of human intelligence."
            ),
            Document(page_content="Machine learning is a subset of AI."),
            Document(page_content="Deep learning uses neural networks."),
        ]
        mock_chroma_db.similarity_search.return_value = mock_documents

        # Mock the graph to simulate the actual flow
        with patch("src.agent.agent_core.graph") as mock_graph:
            mock_graph.stream.return_value = [
                {"question": "What is AI?", "db": mock_chroma_db},
                {"context": mock_documents},
                {"answer": mock_response},
            ]

            result = ask_rag_with_connection("What is AI?", mock_chroma_db)

            assert (
                result
                == "AI is the simulation of human intelligence in machines."
            )


class TestRAGAgentErrorHandling:
    """Test error handling in the RAG agent."""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def test_retrieve_with_empty_results(self, mock_chroma_db):
        """Test retrieve function with empty search results."""
        from src.agent.agent_core import retrieve

        mock_chroma_db.similarity_search.return_value = []

        state = {"question": "What is AI?", "db": mock_chroma_db}
        result = retrieve(state)

        assert result["context"] == []

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    @patch("src.agent.agent_core.llm")
    def test_generate_with_llm_error(self, mock_llm, mock_chroma_db):
        """Test generate function when LLM throws an error."""
        from src.agent.agent_core import generate

        mock_llm.invoke.side_effect = Exception("LLM service unavailable")

        state = {
            "question": "What is AI?",
            "context": [Document(page_content="Test content")],
            "db": mock_chroma_db,
        }

        with pytest.raises(Exception):
            generate(state)

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    @patch("src.agent.agent_core.graph")
    def test_ask_rag_with_graph_error(self, mock_graph, mock_chroma_db):
        """Test ask_rag_with_connection when graph throws an error."""
        from src.agent.agent_core import ask_rag_with_connection

        mock_graph.stream.side_effect = Exception("Graph execution failed")

        with pytest.raises(Exception):
            ask_rag_with_connection("What is AI?", mock_chroma_db)

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def test_missing_api_key(self):
        """Test behavior when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
                # This should raise an error when trying to
                # initialize the module
                import src.agent.agent_core  # noqa: F401


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])
