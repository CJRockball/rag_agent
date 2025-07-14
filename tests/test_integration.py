"""Integration tests for the RAG Chat application."""

import pytest
import os
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from langchain_chroma import Chroma


@pytest.mark.integration
class TestRAGChatIntegration:
    """Integration tests for the complete RAG chat system."""

    @pytest.fixture
    def mock_chroma_db(self):
        """Create a mock Chroma database."""
        mock_db = Mock(spec=Chroma)
        return mock_db

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def test_end_to_end_flow(
        self, test_environment, sample_documents, mock_chroma_db
    ):
        """Test the complete flow from user input to response."""
        with (
            patch("src.agent.agent_core.llm") as mock_llm,
            patch("src.agent.agent_core.graph") as mock_graph,
        ):
            # Mock the LLM response
            mock_response = Mock()
            mock_response.content = (
                "AI is the simulation of human intelligence in machines."
            )
            mock_llm.invoke.return_value = mock_response

            # Mock the graph response
            mock_graph.stream.return_value = [
                {"question": "What is AI?", "db": mock_chroma_db},
                {"context": sample_documents},
                {"answer": mock_response},
            ]

            # Import and test the function
            from src.agent.agent_core import ask_rag_with_connection

            result = ask_rag_with_connection("What is AI?", mock_chroma_db)

            # Verify the complete flow
            assert (
                result
                == "AI is the simulation of human intelligence in machines."
            )
            mock_llm.invoke.assert_called_once()
            mock_graph.stream.assert_called_once()

    @patch("streamlit.secrets")
    @patch("src.utils.security.validate_api_key")
    @patch("src.utils.load_db.setup_vector_database")
    @patch("src.agent.agent_core.ask_rag_with_connection")
    def test_streamlit_integration(
        self,
        mock_ask_rag,
        mock_setup_db,
        mock_validate_api,
        mock_secrets,
        mock_chroma_db,
    ):
        """Test integration between Streamlit app and RAG agent."""
        # Setup mocks
        mock_secrets.__getitem__.side_effect = lambda key: {
            "GOOGLE_API_KEY": "test_api_key",
            "CHROMA_DB_PATH": "test_db_path",
            "COLLECTION_NAME": "test_collection",
            "DOC_PATH": "test_doc_path",
        }[key]
        mock_secrets.get.side_effect = lambda key, default=None: {
            "GOOGLE_API_KEY": "test_api_key",
            "CHROMA_DB_PATH": "test_db_path",
            "COLLECTION_NAME": "test_collection",
            "DOC_PATH": "test_doc_path",
        }.get(key, default)

        mock_setup_db.return_value = mock_chroma_db
        mock_validate_api.return_value = None
        mock_ask_rag.return_value = "Integration test response"

        # Test the integration
        result = mock_ask_rag("Integration test question", mock_chroma_db)
        assert result == "Integration test response"

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def test_error_propagation(self, test_environment, mock_chroma_db):
        """Test that errors are properly propagated through the system."""
        mock_chroma_db.similarity_search.side_effect = Exception(
            "Database error"
        )

        from src.agent.agent_core import retrieve

        with pytest.raises(Exception) as exc_info:
            retrieve({"question": "test", "db": mock_chroma_db})

        assert "Database error" in str(exc_info.value)

    @patch("streamlit.secrets")
    def test_configuration_loading(self, mock_secrets):
        """Test that configuration is properly loaded from \
            Streamlit secrets."""
        mock_secrets.__getitem__.side_effect = lambda key: {
            "GOOGLE_API_KEY": "test_api_key",
            "CHROMA_DB_PATH": "test_db_path",
            "COLLECTION_NAME": "test_collection",
            "DOC_PATH": "test_doc_path",
        }[key]
        mock_secrets.get.side_effect = lambda key, default=None: {
            "GOOGLE_API_KEY": "test_api_key",
            "CHROMA_DB_PATH": "test_db_path",
            "COLLECTION_NAME": "test_collection",
            "DOC_PATH": "test_doc_path",
        }.get(key, default)

        # Mock the environment initialization
        with patch.dict(os.environ, {}, clear=True):
            # Simulate the initialize_environment function
            os.environ["GOOGLE_API_KEY"] = mock_secrets["GOOGLE_API_KEY"]
            os.environ["CHROMA_DB_PATH"] = mock_secrets.get("CHROMA_DB_PATH")
            os.environ["COLLECTION_NAME"] = mock_secrets.get("COLLECTION_NAME")

            # Test that environment variables are set correctly
            assert os.environ.get("GOOGLE_API_KEY") == "test_api_key"
            assert os.environ.get("CHROMA_DB_PATH") == "test_db_path"
            assert os.environ.get("COLLECTION_NAME") == "test_collection"

    @pytest.mark.slow
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def test_performance_under_load(self, performance_monitor, mock_chroma_db):
        """Test system performance under simulated load."""
        performance_monitor.start()

        with patch(
            "src.agent.agent_core.ask_rag_with_connection"
        ) as mock_ask_rag:
            mock_ask_rag.return_value = "Performance test response"

            # Simulate multiple concurrent requests
            for i in range(10):
                result = mock_ask_rag(f"Question {i}", mock_chroma_db)
                assert result == "Performance test response"

        stats = performance_monitor.stop()

        # Performance assertions
        assert stats["duration"] < 1.0  # Should complete within 1 second
        assert stats["memory_used"] < 100 * 1024 * 1024  # Less than 100MB

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def test_data_consistency(self, sample_documents, mock_chroma_db):
        """Test data consistency through the pipeline."""
        with patch("src.agent.agent_core.llm") as mock_llm:
            # Mock the database response
            mock_chroma_db.similarity_search.return_value = sample_documents

            # Mock the LLM response
            mock_response = Mock()
            mock_response.content = "Consistent response"
            mock_llm.invoke.return_value = mock_response

            from src.agent.agent_core import retrieve, generate

            # Test retrieve consistency
            state = {"question": "What is AI?", "db": mock_chroma_db}
            result1 = retrieve(state)
            result2 = retrieve(state)

            assert result1["context"] == result2["context"]

            # Test generate consistency
            state_with_context = {
                "question": "What is AI?",
                "context": sample_documents,
                "db": mock_chroma_db,
            }

            gen_result1 = generate(state_with_context)
            gen_result2 = generate(state_with_context)

            assert gen_result1["answer"] == gen_result2["answer"]


@pytest.mark.integration
class TestSystemResilience:
    """Test system resilience and error handling."""

    @pytest.fixture
    def mock_chroma_db(self):
        """Create a mock Chroma database."""
        mock_db = Mock(spec=Chroma)
        return mock_db

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def test_database_unavailable(self, mock_chroma_db):
        """Test behavior when database is unavailable."""
        mock_chroma_db.similarity_search.side_effect = ConnectionError(
            "Database unavailable"
        )

        from src.agent.agent_core import retrieve

        with pytest.raises(ConnectionError):
            retrieve({"question": "test", "db": mock_chroma_db})

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    @patch("src.agent.agent_core.llm")
    def test_llm_service_unavailable(self, mock_llm, mock_chroma_db):
        """Test behavior when LLM service is unavailable."""
        mock_llm.invoke.side_effect = ConnectionError(
            "LLM service unavailable"
        )

        from src.agent.agent_core import generate

        state = {
            "question": "test",
            "context": [Document(page_content="test content")],
            "db": mock_chroma_db,
        }

        with pytest.raises(ConnectionError):
            generate(state)

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    @patch("src.agent.agent_core.llm")
    def test_partial_system_failure(self, mock_llm, mock_chroma_db):
        """Test system behavior with partial failures."""
        # Database works but returns empty results
        mock_chroma_db.similarity_search.return_value = []

        # LLM works normally
        mock_response = Mock()
        mock_response.content = "Response with no context"
        mock_llm.invoke.return_value = mock_response

        from src.agent.agent_core import retrieve, generate

        # Test retrieve with empty results
        state = {"question": "test", "db": mock_chroma_db}
        result = retrieve(state)
        assert result["context"] == []

        # Test generate with empty context
        state_with_empty_context = {
            "question": "test",
            "context": [],
            "db": mock_chroma_db,
        }

        gen_result = generate(state_with_empty_context)
        assert gen_result["answer"] == mock_response

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    @patch("src.agent.agent_core.llm")
    def test_timeout_handling(self, mock_llm, mock_chroma_db):
        """Test timeout handling in the system."""
        # Simulate timeout
        mock_llm.invoke.side_effect = TimeoutError("Request timed out")

        from src.agent.agent_core import generate

        state = {
            "question": "test",
            "context": [Document(page_content="test content")],
            "db": mock_chroma_db,
        }

        with pytest.raises(TimeoutError):
            generate(state)

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def test_memory_pressure(self, mock_chroma_db):
        """Test system behavior under memory pressure."""
        # Simulate large response
        large_documents = [
            Document(page_content="Large content " * 1000) for _ in range(100)
        ]
        mock_chroma_db.similarity_search.return_value = large_documents

        from src.agent.agent_core import retrieve

        state = {"question": "test", "db": mock_chroma_db}
        result = retrieve(state)

        # Should handle large responses
        assert len(result["context"]) == 100
        assert all(isinstance(doc, Document) for doc in result["context"])


@pytest.mark.integration
class TestDataFlow:
    """Test data flow through the system."""

    @pytest.fixture
    def mock_chroma_db(self):
        """Create a mock Chroma database."""
        mock_db = Mock(spec=Chroma)
        return mock_db

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    def test_question_preprocessing(self, mock_chroma_db):
        """Test question preprocessing and handling."""
        test_questions = [
            "What is AI?",
            "what is ai?",
            "WHAT IS AI?",
            "What is AI???",
            " What is AI? ",
            "",
        ]

        mock_chroma_db.similarity_search.return_value = []

        from src.agent.agent_core import retrieve

        for question in test_questions:
            state = {"question": question, "db": mock_chroma_db}
            result = retrieve(state)

            # Should handle all question formats
            assert "context" in result
            mock_chroma_db.similarity_search.assert_called_with(
                query=question, k=3
            )

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    @patch("src.agent.agent_core.llm")
    def test_context_processing(
        self, mock_llm, sample_documents, mock_chroma_db
    ):
        """Test context processing and formatting."""
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_llm.invoke.return_value = mock_response

        # Check that LLM was called with properly formatted context
        call_args = mock_llm.invoke.call_args[0][0]
        system_message = call_args[0]

        # Should contain context from all documents
        assert "Artificial Intelligence" in system_message.content
        assert "Machine Learning" in system_message.content
        assert "Deep Learning" in system_message.content

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"})
    @patch("src.agent.agent_core.graph")
    def test_answer_postprocessing(self, mock_graph, mock_chroma_db):
        """Test answer postprocessing and formatting."""
        test_responses = [
            Mock(content="Normal response"),
            Mock(content=""),
            Mock(content="Very long response " * 100),
            "String response",
            None,
        ]

        from src.agent.agent_core import ask_rag_with_connection

        for response in test_responses:
            mock_graph.stream.return_value = [
                {"question": "test", "db": mock_chroma_db},
                {"answer": response},
            ]

            result = ask_rag_with_connection("test", mock_chroma_db)

            # Should handle all response types
            if response is None:
                assert result is None
            elif hasattr(response, "content"):
                assert result == response.content
            else:
                assert result == response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
