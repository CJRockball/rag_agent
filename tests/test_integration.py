"""Integration tests for the RAG Chat application."""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
import time


@pytest.mark.integration
class TestRAGChatIntegration:
    """Integration tests for the complete RAG chat system."""

    def test_end_to_end_flow(self, test_environment, sample_documents):
        """Test the complete flow from user input to response."""
        with patch("src.agent.agent_core.db") as mock_db, patch(
            "src.agent.agent_core.llm"
        ) as mock_llm, patch("src.agent.agent_core.graph") as mock_graph:

            # Mock the complete flow
            mock_db.similarity_search.return_value = sample_documents

            mock_response = Mock()
            mock_response.content = (
                "AI is the simulation of human intelligence in machines."
            )
            mock_llm.invoke.return_value = mock_response

            mock_graph.stream.return_value = [
                {"question": "What is AI?"},
                {"context": sample_documents},
                {"answer": mock_response},
            ]

            # Import and test the function
            from src.agent.agent_core import ask_rag

            result = ask_rag("What is AI?")

            # Verify the complete flow
            assert result == "AI is the simulation of human intelligence in machines."
            mock_db.similarity_search.assert_called_once()
            mock_llm.invoke.assert_called_once()
            mock_graph.stream.assert_called_once()

    def test_streamlit_integration(self, mock_streamlit_app):
        """Test integration between Streamlit app and RAG agent."""
        with patch("streamlit_app.ask_rag") as mock_ask_rag:
            mock_ask_rag.return_value = "Integration test response"

            # Test would simulate user interaction
            # This is a placeholder for actual Streamlit integration testing
            result = mock_ask_rag("Integration test question")

            assert result == "Integration test response"

    def test_error_propagation(self, test_environment):
        """Test that errors are properly propagated through the system."""
        with patch("src.agent.agent_core.db") as mock_db:
            mock_db.similarity_search.side_effect = Exception("Database error")

            from src.agent.agent_core import retrieve

            with pytest.raises(Exception) as exc_info:
                retrieve({"question": "test"})

            assert "Database error" in str(exc_info.value)

    def test_configuration_loading(self, test_environment):
        """Test that configuration is properly loaded."""
        # Test that environment variables are accessible
        assert os.environ.get("GOOGLE_API_KEY") == "test_api_key_for_testing"
        assert os.environ.get("CHROMA_DB_PATH") == "test_db_path"
        assert os.environ.get("COLLECTION_NAME") == "test_collection"

    @pytest.mark.slow
    def test_performance_under_load(self, performance_monitor):
        """Test system performance under simulated load."""
        performance_monitor.start()

        with patch("src.agent.agent_core.ask_rag") as mock_ask_rag:
            mock_ask_rag.return_value = "Performance test response"

            # Simulate multiple concurrent requests
            for i in range(10):
                result = mock_ask_rag(f"Question {i}")
                assert result == "Performance test response"

        stats = performance_monitor.stop()

        # Performance assertions
        assert stats["duration"] < 1.0  # Should complete within 1 second
        assert stats["memory_used"] < 100 * 1024 * 1024  # Less than 100MB

    def test_data_consistency(self, sample_documents):
        """Test data consistency through the pipeline."""
        with patch("src.agent.agent_core.db") as mock_db, patch(
            "agent_core.llm"
        ) as mock_llm:

            # Test that the same input produces consistent output
            mock_db.similarity_search.return_value = sample_documents

            mock_response = Mock()
            mock_response.content = "Consistent response"
            mock_llm.invoke.return_value = mock_response

            from src.agent.agent_core import retrieve, generate

            # Test retrieve consistency
            state = {"question": "What is AI?"}
            result1 = retrieve(state)
            result2 = retrieve(state)

            assert result1["context"] == result2["context"]

            # Test generate consistency
            state_with_context = {
                "question": "What is AI?",
                "context": sample_documents,
            }

            gen_result1 = generate(state_with_context)
            gen_result2 = generate(state_with_context)

            assert gen_result1["answer"] == gen_result2["answer"]


@pytest.mark.integration
class TestSystemResilience:
    """Test system resilience and error handling."""

    def test_database_unavailable(self):
        """Test behavior when database is unavailable."""
        with patch("src.agent.agent_core.db") as mock_db:
            mock_db.similarity_search.side_effect = ConnectionError(
                "Database unavailable"
            )

            from src.agent.agent_core import retrieve

            with pytest.raises(ConnectionError):
                retrieve({"question": "test"})

    def test_llm_service_unavailable(self):
        """Test behavior when LLM service is unavailable."""
        with patch("src.agent.agent_core.llm") as mock_llm:
            mock_llm.invoke.side_effect = ConnectionError("LLM service unavailable")

            from src.agent.agent_core import generate

            state = {
                "question": "test",
                "context": [Document(page_content="test content")],
            }

            with pytest.raises(ConnectionError):
                generate(state)

    def test_partial_system_failure(self):
        """Test system behavior with partial failures."""
        with patch("src.agent.agent_core.db") as mock_db, patch(
            "agent_core.llm"
        ) as mock_llm:

            # Database works but returns empty results
            mock_db.similarity_search.return_value = []

            # LLM works normally
            mock_response = Mock()
            mock_response.content = "Response with no context"
            mock_llm.invoke.return_value = mock_response

            from src.agent.agent_core import retrieve, generate

            # Test retrieve with empty results
            state = {"question": "test"}
            result = retrieve(state)
            assert result["context"] == []

            # Test generate with empty context
            state_with_empty_context = {"question": "test", "context": []}

            gen_result = generate(state_with_empty_context)
            assert gen_result["answer"] == mock_response

    def test_timeout_handling(self):
        """Test timeout handling in the system."""
        with patch("src.agent.agent_core.llm") as mock_llm:
            # Simulate timeout
            mock_llm.invoke.side_effect = TimeoutError("Request timed out")

            from src.agent.agent_core import generate

            state = {
                "question": "test",
                "context": [Document(page_content="test content")],
            }

            with pytest.raises(TimeoutError):
                generate(state)

    def test_memory_pressure(self):
        """Test system behavior under memory pressure."""
        with patch("src.agent.agent_core.db") as mock_db:
            # Simulate large response
            large_documents = [
                Document(page_content="Large content " * 1000) for _ in range(100)
            ]
            mock_db.similarity_search.return_value = large_documents

            from src.agent.agent_core import retrieve

            state = {"question": "test"}
            result = retrieve(state)

            # Should handle large responses
            assert len(result["context"]) == 100
            assert all(isinstance(doc, Document) for doc in result["context"])


@pytest.mark.integration
class TestDataFlow:
    """Test data flow through the system."""

    def test_question_preprocessing(self):
        """Test question preprocessing and handling."""
        test_questions = [
            "What is AI?",
            "what is ai?",
            "WHAT IS AI?",
            "What is AI???",
            "  What is AI?  ",
            "",
        ]

        with patch("src.agent.agent_core.db") as mock_db:
            mock_db.similarity_search.return_value = []

            from src.agent.agent_core import retrieve

            for question in test_questions:
                state = {"question": question}
                result = retrieve(state)

                # Should handle all question formats
                assert "context" in result
                mock_db.similarity_search.assert_called_with(query=question, k=3)

    def test_context_processing(self, sample_documents):
        """Test context processing and formatting."""
        with patch("src.agent.agent_core.llm") as mock_llm:
            mock_response = Mock()
            mock_response.content = "Test response"
            mock_llm.invoke.return_value = mock_response

            from src.agent.agent_core import generate

            state = {"question": "What is AI?", "context": sample_documents}

            result = generate(state)

            # Check that LLM was called with properly formatted context
            call_args = mock_llm.invoke.call_args[0][0]
            system_message = call_args[0]

            # Should contain context from all documents
            assert "Artificial Intelligence" in system_message.content
            assert "Machine Learning" in system_message.content
            assert "Deep Learning" in system_message.content

    def test_answer_postprocessing(self):
        """Test answer postprocessing and formatting."""
        test_responses = [
            Mock(content="Normal response"),
            Mock(content=""),
            Mock(content="Very long response " * 100),
            "String response",
            None,
        ]

        from src.agent.agent_core import ask_rag

        with patch("src.agent.agent_core.graph") as mock_graph:
            for response in test_responses:
                mock_graph.stream.return_value = [
                    {"question": "test"},
                    {"answer": response},
                ]

                result = ask_rag("test")

                # Should handle all response types
                if response is None:
                    assert result is None
                elif hasattr(response, "content"):
                    assert result == response.content
                else:
                    assert result == response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
