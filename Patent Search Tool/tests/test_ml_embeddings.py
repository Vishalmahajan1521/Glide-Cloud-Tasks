"""
Tests for ML embeddings
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.ml.embeddings import EmbeddingModel


class TestEmbeddingModel:
    """Tests for EmbeddingModel class"""
    
    @patch('app.ml.embeddings.settings')
    @patch('app.ml.embeddings.requests')
    def test_embed_single_success(self, mock_requests, mock_settings):
        """Test embedding a single text successfully"""
        # Setup mocks
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.ollama_model = "nomic-embed-text"
        
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1] * 768}
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        # Create model and test
        model = EmbeddingModel()
        result = model._embed_single("test text")
        
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)
        mock_requests.post.assert_called_once()
    
    @patch('app.ml.embeddings.settings')
    @patch('app.ml.embeddings.requests')
    def test_embed_single_error(self, mock_requests, mock_settings):
        """Test embedding when API returns error"""
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.ollama_model = "nomic-embed-text"
        
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_requests.post.return_value = mock_response
        
        model = EmbeddingModel()
        
        with pytest.raises(Exception):
            model._embed_single("test text")
    
    @patch('app.ml.embeddings.settings')
    @patch('app.ml.embeddings.requests')
    def test_embed_documents(self, mock_requests, mock_settings):
        """Test embedding multiple documents"""
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.ollama_model = "nomic-embed-text"
        
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1] * 768}
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        model = EmbeddingModel()
        texts = ["text1", "text2", "text3"]
        results = model.embed_documents(texts)
        
        assert len(results) == 3
        assert all(len(emb) == 768 for emb in results)
        assert mock_requests.post.call_count == 3
    
    @patch('app.ml.embeddings.settings')
    @patch('app.ml.embeddings.requests')
    def test_embed_query(self, mock_requests, mock_settings):
        """Test embedding a query"""
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.ollama_model = "nomic-embed-text"
        
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.2] * 768}
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        model = EmbeddingModel()
        result = model.embed_query("search query")
        
        assert len(result) == 768
        mock_requests.post.assert_called_once()
    
    @patch('app.ml.embeddings.settings')
    @patch('app.ml.embeddings.requests')
    def test_embed_request_payload(self, mock_requests, mock_settings):
        """Test that embedding request has correct payload"""
        mock_settings.ollama_url = "http://localhost:11434"
        mock_settings.ollama_model = "test-model"
        
        mock_response = Mock()
        mock_response.json.return_value = {"embedding": [0.1] * 768}
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        model = EmbeddingModel()
        model._embed_single("test text")
        
        # Verify request was made with correct parameters
        call_args = mock_requests.post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/embeddings"
        assert call_args[1]["json"]["model"] == "test-model"
        assert call_args[1]["json"]["prompt"] == "test text"
        assert call_args[1]["timeout"] == 60
