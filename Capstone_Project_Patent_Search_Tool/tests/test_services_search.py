"""
Tests for SearchService
"""
import pytest
from unittest.mock import Mock, MagicMock
from app.services.search_service import SearchService
from app.models.schemas.search import SearchRequest, SearchFilters
from app.core.exceptions import SearchError


class TestSearchService:
    """Tests for SearchService class"""
    
    def test_init(self):
        """Test SearchService initialization"""
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        service = SearchService(mock_vector_store, mock_embedder)
        
        assert service.vector_store == mock_vector_store
        assert service.embedder == mock_embedder
    
    def test_search_success(self):
        """Test successful search"""
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        # Setup mocks
        mock_embedder.embed_query.return_value = [0.1] * 768
        
        mock_point1 = Mock()
        mock_point1.score = 0.95
        mock_point1.payload = {
            "text": "Sample text 1",
            "patent_id": "US12345678",
            "title": "Test Patent 1",
            "assignee": "Company A",
            "jurisdiction": "US",
            "filing_year": 2020,
            "patent_class": ["H01M"],
            "chunk_type": "claim"
        }
        
        mock_point2 = Mock()
        mock_point2.score = 0.85
        mock_point2.payload = {
            "text": "Sample text 2",
            "patent_id": "US87654321",
            "title": "Test Patent 2",
            "assignee": "Company B",
            "jurisdiction": "EP",
            "filing_year": 2019,
            "patent_class": ["B60L"],
            "chunk_type": "abstract"
        }
        
        mock_results = Mock()
        mock_results.points = [mock_point1, mock_point2]
        mock_vector_store.search.return_value = mock_results
        
        service = SearchService(mock_vector_store, mock_embedder)
        
        request = SearchRequest(query="battery technology", top_k=10)
        results = service.search(request)
        
        # Verify embedder was called
        mock_embedder.embed_query.assert_called_once_with("battery technology")
        
        # Verify vector store was called
        mock_vector_store.search.assert_called_once()
        
        # Verify results
        assert len(results) == 2
        assert results[0]["score"] == 0.95
        assert results[0]["patent_id"] == "US12345678"
        assert results[0]["text"] == "Sample text 1"
        assert results[1]["score"] == 0.85
    
    def test_search_with_filters(self):
        """Test search with filters"""
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        mock_embedder.embed_query.return_value = [0.1] * 768
        
        mock_results = Mock()
        mock_results.points = []
        mock_vector_store.search.return_value = mock_results
        
        service = SearchService(mock_vector_store, mock_embedder)
        
        filters = SearchFilters(jurisdiction=["US"], filing_year_from=2020)
        request = SearchRequest(
            query="battery technology",
            top_k=5,
            filters=filters
        )
        
        results = service.search(request)
        
        # Verify filters were passed to vector store
        call_args = mock_vector_store.search.call_args
        assert call_args[1]["filters"] == filters
    
    def test_search_error(self):
        """Test search error handling"""
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        mock_embedder.embed_query.side_effect = Exception("Embedding error")
        
        service = SearchService(mock_vector_store, mock_embedder)
        
        request = SearchRequest(query="test query")
        
        with pytest.raises(SearchError):
            service.search(request)
    
    def test_search_vector_store_error(self):
        """Test search when vector store raises error"""
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_vector_store.search.side_effect = Exception("Qdrant error")
        
        service = SearchService(mock_vector_store, mock_embedder)
        
        request = SearchRequest(query="test query")
        
        with pytest.raises(SearchError):
            service.search(request)
    
    def test_build_explanation_claim(self):
        """Test building explanation for claim chunk"""
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        service = SearchService(mock_vector_store, mock_embedder)
        
        mock_hit = Mock()
        mock_hit.score = 0.95
        mock_hit.payload = {"chunk_type": "claim"}
        
        explanation = service._build_explanation(mock_hit)
        
        assert "claim" in explanation.lower()
        assert "0.95" in explanation or "0.9500" in explanation
    
    def test_build_explanation_abstract(self):
        """Test building explanation for abstract chunk"""
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        service = SearchService(mock_vector_store, mock_embedder)
        
        mock_hit = Mock()
        mock_hit.score = 0.85
        mock_hit.payload = {"chunk_type": "abstract"}
        
        explanation = service._build_explanation(mock_hit)
        
        assert "abstract" in explanation.lower()
    
    def test_build_explanation_description(self):
        """Test building explanation for description chunk"""
        mock_vector_store = Mock()
        mock_embedder = Mock()
        
        service = SearchService(mock_vector_store, mock_embedder)
        
        mock_hit = Mock()
        mock_hit.score = 0.75
        mock_hit.payload = {"chunk_type": "description"}
        
        explanation = service._build_explanation(mock_hit)
        
        assert "description" in explanation.lower()
