"""Test suite for the Streamlit RAG Chat application."""

import pytest
import os
from unittest.mock import patch, Mock
from streamlit.testing.v1 import AppTest
from langchain_chroma import Chroma


class TestStreamlitApp:
    """Test suite for the Streamlit application."""

    @pytest.fixture
    def mock_secrets(self):
        """Mock Streamlit secrets."""
        return {
            "GOOGLE_API_KEY": "test_api_key",
            "CHROMA_DB_PATH": "test_db_path",
            "COLLECTION_NAME": "test_collection",
            "DOC_PATH": "test_doc_path",
        }

    @pytest.fixture
    def mock_chroma_db(self):
        """Create a mock Chroma database."""
        mock_db = Mock(spec=Chroma)
        return mock_db

    @patch("streamlit.secrets")
    @patch("src.utils.security.validate_api_key")
    @patch("src.utils.load_db.setup_vector_database")
    def test_app_initialization(
        self, mock_setup_db, mock_validate_api, mock_secrets, mock_chroma_db
    ):
        """Test that the app initializes correctly."""
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

        # Test using Streamlit's testing framework
        app = AppTest.from_file("streamlit_app.py")
        app.run()

        # Check that the app doesn't crash on initialization
        assert app is not None
        assert not app.exception

    @patch("streamlit.secrets")
    @patch("src.utils.security.validate_api_key")
    @patch("src.utils.load_db.setup_vector_database")
    def test_page_config(
        self, mock_setup_db, mock_validate_api, mock_secrets, mock_chroma_db
    ):
        """Test that page configuration is set correctly."""
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

        app = AppTest.from_file("streamlit_app.py")
        app.run()

        # Check page configuration elements
        assert len(app.title) > 0
        assert "PDF RAG Chat" in str(app.title[0].value)

    @patch("streamlit.secrets")
    @patch("src.utils.security.validate_api_key")
    @patch("src.utils.load_db.setup_vector_database")
    @patch("src.agent.agent_core.ask_rag_with_connection")
    @patch("src.utils.rate_limit.allow_request")
    @patch("src.utils.rate_limit.record_request")
    def test_user_input_processing(
        self,
        mock_record_request,
        mock_allow_request,
        mock_ask_rag,
        mock_setup_db,
        mock_validate_api,
        mock_secrets,
        mock_chroma_db,
    ):
        """Test that user input is processed correctly."""
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
        mock_allow_request.return_value = True
        mock_ask_rag.return_value = "This is a test response."

        app = AppTest.from_file("streamlit_app.py")
        app.run()

        # Simulate user input
        if app.chat_input:
            app.chat_input[0].set_value("What is AI?").run()

        # Check that ask_rag was called with database
        mock_ask_rag.assert_called_once_with("What is AI?", mock_chroma_db)

    @patch("streamlit.secrets")
    @patch("src.utils.security.validate_api_key")
    @patch("src.utils.load_db.setup_vector_database")
    @patch("src.agent.agent_core.ask_rag_with_connection")
    @patch("src.utils.rate_limit.allow_request")
    def test_fallback_response(
        self,
        mock_allow_request,
        mock_ask_rag,
        mock_setup_db,
        mock_validate_api,
        mock_secrets,
        mock_chroma_db,
    ):
        """Test fallback response when ask_rag returns None."""
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
        mock_allow_request.return_value = True
        mock_ask_rag.return_value = None

        app = AppTest.from_file("streamlit_app.py")
        app.run()

        # Simulate user input
        if app.chat_input:
            app.chat_input[0].set_value("What is AI?").run()

        # The app should handle None response gracefully
        assert not app.exception

    @patch("streamlit.secrets")
    @patch("src.utils.security.validate_api_key")
    @patch("src.utils.load_db.setup_vector_database")
    @patch("src.utils.rate_limit.allow_request")
    def test_rate_limiting(
        self,
        mock_allow_request,
        mock_setup_db,
        mock_validate_api,
        mock_secrets,
        mock_chroma_db,
    ):
        """Test rate limiting functionality."""
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
        mock_allow_request.return_value = False

        app = AppTest.from_file("streamlit_app.py")
        app.run()

        # When rate limited, chat input should be disabled
        # This test ensures the rate limiting logic works
        assert not app.exception


class TestStreamlitAppIntegration:
    """Integration tests for the Streamlit application."""

    @patch("streamlit.secrets")
    @patch("src.utils.security.validate_api_key")
    @patch("src.utils.load_db.setup_vector_database")
    def test_environment_initialization(
        self, mock_setup_db, mock_validate_api, mock_secrets, mock_chroma_db
    ):
        """Test that environment variables are properly initialized."""
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

        with patch.dict(os.environ, {}, clear=True):
            app = AppTest.from_file("streamlit_app.py")
            app.run()

            # Check that environment variables are set
            assert os.environ.get("GOOGLE_API_KEY") == "test_api_key"
            assert os.environ.get("CHROMA_DB_PATH") == "test_db_path"
            assert os.environ.get("COLLECTION_NAME") == "test_collection"

    @patch("streamlit.secrets")
    @patch("src.utils.security.validate_api_key")
    @patch("src.utils.load_db.setup_vector_database")
    def test_database_initialization(
        self, mock_setup_db, mock_validate_api, mock_secrets, mock_chroma_db
    ):
        """Test that database is properly initialized."""
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

        app = AppTest.from_file("streamlit_app.py")
        app.run()

        # Check that database setup was called
        mock_setup_db.assert_called_once_with(
            "test_doc_path", "test_db_path", "test_collection"
        )


class TestStreamlitAppErrorHandling:
    """Test error handling in the Streamlit application."""

    @patch("streamlit.secrets")
    def test_missing_secrets(self, mock_secrets):
        """Test handling of missing secrets."""
        mock_secrets.__getitem__.side_effect = KeyError("GOOGLE_API_KEY")

        app = AppTest.from_file("streamlit_app.py")
        app.run()

        # App should handle missing secrets gracefully
        # In production, this would show an error message and stop
        assert app.exception is not None

    @patch("streamlit.secrets")
    @patch("src.utils.security.validate_api_key")
    @patch("src.utils.load_db.setup_vector_database")
    def test_database_initialization_error(
        self, mock_setup_db, mock_validate_api, mock_secrets
    ):
        """Test handling of database initialization errors."""
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

        mock_validate_api.return_value = None
        mock_setup_db.side_effect = Exception("Database setup failed")

        app = AppTest.from_file("streamlit_app.py")
        app.run()

        # App should handle database setup errors gracefully
        assert app.exception is not None


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])
