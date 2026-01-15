"""
Tests for QdrantStore
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4
from app.retrieval.qdrant_store import QdrantStore


class TestQdrantStore:
    """Tests for QdrantStore class"""
    
    @patch('app.retrieval.qdrant_store.QdrantClient')
    @patch('app.retrieval.qdrant_store.settings')
    def test_init(self, mock_settings, mock_qdrant_client):
        """Test QdrantStore initialization"""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        
        store = QdrantStore()
        
        assert store.collection_name == "patent_chunks"
        mock_qdrant_client.assert_called_once_with(
            host="localhost",
            port=6333
        )
    
    @patch('app.retrieval.qdrant_store.QdrantClient')
    @patch('app.retrieval.qdrant_store.settings')
    def test_create_collection_new(self, mock_settings, mock_qdrant_client):
        """Test creating a new collection"""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        
        mock_client_instance = Mock()
        mock_client_instance.get_collections.return_value = Mock(collections=[])
        mock_qdrant_client.return_value = mock_client_instance
        
        store = QdrantStore()
        store.create_collection()
        
        # Should call recreate_collection
        mock_client_instance.recreate_collection.assert_called_once()
        # Should create payload indexes
        assert mock_client_instance.create_payload_index.call_count == 6
    
    @patch('app.retrieval.qdrant_store.QdrantClient')
    @patch('app.retrieval.qdrant_store.settings')
    def test_create_collection_exists(self, mock_settings, mock_qdrant_client):
        """Test creating collection when it already exists"""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        
        mock_client_instance = Mock()
        existing_collection = Mock()
        existing_collection.name = "patent_chunks"
        mock_client_instance.get_collections.return_value = Mock(
            collections=[existing_collection]
        )
        mock_qdrant_client.return_value = mock_client_instance
        
        store = QdrantStore()
        store.create_collection()
        
        # Should not call recreate_collection
        mock_client_instance.recreate_collection.assert_not_called()
    
    @patch('app.retrieval.qdrant_store.QdrantClient')
    @patch('app.retrieval.qdrant_store.settings')
    def test_delete_collection(self, mock_settings, mock_qdrant_client):
        """Test deleting a collection"""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        
        store = QdrantStore()
        store.delete_collection()
        
        mock_client_instance.delete_collection.assert_called_once_with("patent_chunks")
    
    @patch('app.retrieval.qdrant_store.QdrantClient')
    @patch('app.retrieval.qdrant_store.settings')
    def test_upsert_chunks(self, mock_settings, mock_qdrant_client):
        """Test upserting chunks"""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        
        mock_client_instance = Mock()
        mock_qdrant_client.return_value = mock_client_instance
        
        store = QdrantStore()
        
        chunks = [
            {
                "text": "chunk1",
                "chunk_type": "abstract",
                "section_priority": 0.7,
                "chunk_index": 0
            },
            {
                "text": "chunk2",
                "chunk_type": "claim",
                "section_priority": 1.0,
                "claim_number": 1,
                "chunk_index": 0
            }
        ]
        
        embeddings = [[0.1] * 768, [0.2] * 768]
        metadata = {
            "patent_id": "US12345678",
            "title": "Test Patent",
            "assignee": "Test Company",
            "jurisdiction": "US",
            "filing_year": 2020,
            "patent_class": ["H01M"]
        }
        
        store.upsert_chunks(chunks, embeddings, metadata)
        
        # Verify upsert was called
        mock_client_instance.upsert.assert_called_once()
        call_args = mock_client_instance.upsert.call_args
        
        assert call_args[1]["collection_name"] == "patent_chunks"
        points = call_args[1]["points"]
        assert len(points) == 2
        
        # Verify point structure
        for point in points:
            assert "id" in point
            assert "vector" in point
            assert "payload" in point
            assert "text" in point["payload"]
            assert "chunk_type" in point["payload"]
    
    @patch('app.retrieval.qdrant_store.QdrantClient')
    @patch('app.retrieval.qdrant_store.settings')
    def test_search_no_filters(self, mock_settings, mock_qdrant_client):
        """Test search without filters"""
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        
        mock_client_instance = Mock()
        mock_result = Mock()
        mock_result.points = []
        mock_client_instance.query_points.return_value = mock_result
        mock_qdrant_client.return_value = mock_client_instance
        
        store = QdrantStore()
        query_vector = [0.1] * 768
        
        result = store.search(query_vector, top_k=10, filters=None)
        
        mock_client_instance.query_points.assert_called_once()
        call_args = mock_client_instance.query_points.call_args
        
        assert call_args[1]["collection_name"] == "patent_chunks"
        assert call_args[1]["query"] == query_vector
        assert call_args[1]["limit"] == 10
        assert call_args[1]["query_filter"] is None
    
    @patch('app.retrieval.qdrant_store.QdrantClient')
    @patch('app.retrieval.qdrant_store.settings')
    def test_search_with_filters(self, mock_settings, mock_qdrant_client):
        """Test search with filters"""
        from app.models.schemas.search import SearchFilters
        
        mock_settings.qdrant_host = "localhost"
        mock_settings.qdrant_port = 6333
        
        mock_client_instance = Mock()
        mock_result = Mock()
        mock_result.points = []
        mock_client_instance.query_points.return_value = mock_result
        mock_qdrant_client.return_value = mock_client_instance
        
        store = QdrantStore()
        query_vector = [0.1] * 768
        
        filters = SearchFilters(
            jurisdiction=["US", "EP"],
            assignee=["Company A"],
            filing_year_from=2015,
            filing_year_to=2020,
            patent_class=["H01M"],
            topic="thermal_management"
        )
        
        result = store.search(query_vector, top_k=5, filters=filters)
        
        # Verify filter was created
        call_args = mock_client_instance.query_points.call_args
        assert call_args[1]["query_filter"] is not None
