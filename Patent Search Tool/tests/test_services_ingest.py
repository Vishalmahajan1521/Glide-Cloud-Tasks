"""
Tests for IngestService
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.ingest_service import IngestService
from app.core.exceptions import IngestionError


class TestIngestService:
    """Tests for IngestService class"""
    
    @patch('app.services.ingest_service.QdrantStore')
    def test_init(self, mock_qdrant_store):
        """Test IngestService initialization"""
        service = IngestService()
        
        assert service.vector_store is not None
        mock_qdrant_store.assert_called_once()
    
    @patch('app.services.ingest_service.split_into_sections')
    @patch('app.services.ingest_service.create_chunks')
    @patch('app.services.ingest_service.embedding_model')
    @patch('app.services.ingest_service.QdrantStore')
    def test_ingest_patent_success(
        self,
        mock_qdrant_store,
        mock_embedding_model,
        mock_create_chunks,
        mock_split_sections
    ):
        """Test successful patent ingestion from PDF"""
        # Setup mocks
        mock_split_sections.return_value = {"abstract": "Abstract text"}
        
        chunks = [
            {
                "text": "chunk1",
                "chunk_type": "abstract",
                "section_priority": 0.7,
                "chunk_index": 0
            }
        ]
        mock_create_chunks.return_value = chunks
        
        mock_embedding_model.embed_documents.return_value = [[0.1] * 768]
        
        mock_store_instance = Mock()
        mock_qdrant_store.return_value = mock_store_instance
        
        service = IngestService()
        service.vector_store = mock_store_instance
        
        # Mock extract_text_from_pdf at the point of use
        with patch('app.services.ingest_service.extract_text_from_pdf', return_value="Sample patent text", create=True):
            metadata = {
                "patent_id": "US12345678",
                "title": "Test Patent",
                "assignee": "Test Company",
                "jurisdiction": "US",
                "filing_year": 2020,
                "patent_class": ["H01M"]
            }
            
            result = service.ingest_patent("test.pdf", metadata, topic="thermal_management")
        
        assert result["status"] == "success"
        assert result["patent_id"] == "US12345678"
        assert result["chunks_created"] == 1
        
        # Verify embeddings were generated
        mock_embedding_model.embed_documents.assert_called_once()
        
        # Verify chunks were upserted
        mock_store_instance.upsert_chunks.assert_called_once()
    
    @patch('app.services.ingest_service.split_into_sections')
    @patch('app.services.ingest_service.create_chunks')
    @patch('app.services.ingest_service.QdrantStore')
    def test_ingest_patent_no_chunks(
        self,
        mock_qdrant_store,
        mock_create_chunks,
        mock_split_sections
    ):
        """Test ingestion when no chunks are created"""
        mock_split_sections.return_value = {}
        mock_create_chunks.return_value = []
        
        mock_store_instance = Mock()
        mock_qdrant_store.return_value = mock_store_instance
        
        service = IngestService()
        service.vector_store = mock_store_instance
        
        metadata = {"patent_id": "US12345678"}
        
        with patch('app.services.ingest_service.extract_text_from_pdf', return_value="Sample text", create=True):
            with pytest.raises(IngestionError):
                service.ingest_patent("test.pdf", metadata)
    
    @patch('app.utils.patent_api_client.fetch_patent_data')
    @patch('app.services.ingest_service.split_into_sections')
    @patch('app.services.ingest_service.create_chunks')
    @patch('app.services.ingest_service.embedding_model')
    @patch('app.services.ingest_service.QdrantStore')
    def test_ingest_from_api_success(
        self,
        mock_qdrant_store,
        mock_embedding_model,
        mock_create_chunks,
        mock_split_sections,
        mock_fetch_patent
    ):
        """Test successful ingestion from API"""
        mock_fetch_patent.return_value = {
            "text": "Sample patent text",
            "metadata": {
                "patent_id": "US12345678",
                "title": "Test Patent",
                "assignee": "Test Company",
                "jurisdiction": "US",
                "filing_year": 2020,
                "patent_class": ["H01M"]
            }
        }
        
        mock_split_sections.return_value = {"abstract": "Abstract text"}
        chunks = [{"text": "chunk1", "chunk_type": "abstract", "section_priority": 0.7, "chunk_index": 0}]
        mock_create_chunks.return_value = chunks
        mock_embedding_model.embed_documents.return_value = [[0.1] * 768]
        
        mock_store_instance = Mock()
        mock_qdrant_store.return_value = mock_store_instance
        
        service = IngestService()
        service.vector_store = mock_store_instance
        
        result = service.ingest_from_api("US12345678", topic="thermal_management")
        
        assert result["status"] == "success"
        assert result["patent_id"] == "US12345678"
        assert result["chunks_created"] == 1
    
    @patch('app.utils.patent_api_client.fetch_patent_data')
    @patch('app.services.ingest_service.QdrantStore')
    def test_ingest_from_api_no_text(
        self,
        mock_qdrant_store,
        mock_fetch_patent
    ):
        """Test ingestion from API when no text is returned"""
        mock_fetch_patent.return_value = {
            "text": "",
            "metadata": {"patent_id": "US12345678"}
        }
        
        mock_store_instance = Mock()
        mock_qdrant_store.return_value = mock_store_instance
        
        service = IngestService()
        service.vector_store = mock_store_instance
        
        with pytest.raises(IngestionError):
            service.ingest_from_api("US12345678")
    
    @patch('app.services.ingest_service.split_into_sections')
    @patch('app.services.ingest_service.create_chunks')
    @patch('app.services.ingest_service.embedding_model')
    @patch('app.services.ingest_service.QdrantStore')
    def test_ingest_from_text_success(
        self,
        mock_qdrant_store,
        mock_embedding_model,
        mock_create_chunks,
        mock_split_sections
    ):
        """Test successful ingestion from text"""
        mock_split_sections.return_value = {"abstract": "Abstract text"}
        chunks = [{"text": "chunk1", "chunk_type": "abstract", "section_priority": 0.7, "chunk_index": 0}]
        mock_create_chunks.return_value = chunks
        mock_embedding_model.embed_documents.return_value = [[0.1] * 768]
        
        mock_store_instance = Mock()
        mock_qdrant_store.return_value = mock_store_instance
        
        service = IngestService()
        service.vector_store = mock_store_instance
        
        metadata = {
            "patent_id": "US12345678",
            "title": "Test Patent",
            "assignee": "Test Company",
            "jurisdiction": "US",
            "filing_year": 2020,
            "patent_class": ["H01M"]
        }
        
        result = service.ingest_from_text("Sample text", metadata, topic="thermal_management")
        
        assert result["status"] == "success"
        assert result["patent_id"] == "US12345678"
        
        # Verify chunk_type was overridden to "abstract"
        upsert_call = mock_store_instance.upsert_chunks.call_args
        chunks_passed = upsert_call[1]["chunks"]
        assert all(chunk["chunk_type"] == "abstract" for chunk in chunks_passed)
    
    @patch('app.services.ingest_service.split_into_sections')
    @patch('app.services.ingest_service.create_chunks')
    @patch('app.services.ingest_service.QdrantStore')
    def test_ingest_from_text_no_chunks(
        self,
        mock_qdrant_store,
        mock_create_chunks,
        mock_split_sections
    ):
        """Test ingestion from text when no chunks are created"""
        mock_split_sections.return_value = {}
        mock_create_chunks.return_value = []
        
        mock_store_instance = Mock()
        mock_qdrant_store.return_value = mock_store_instance
        
        service = IngestService()
        service.vector_store = mock_store_instance
        
        metadata = {"patent_id": "US12345678"}
        
        result = service.ingest_from_text("", metadata)
        
        assert result["status"] == "skipped"
        assert result["reason"] == "no chunks created"
    
    @patch('app.services.ingest_service.QdrantStore')
    def test_ingest_patent_error(
        self,
        mock_qdrant_store
    ):
        """Test ingestion error handling"""
        mock_store_instance = Mock()
        mock_qdrant_store.return_value = mock_store_instance
        
        service = IngestService()
        service.vector_store = mock_store_instance
        
        metadata = {"patent_id": "US12345678"}
        
        with patch('app.services.ingest_service.extract_text_from_pdf', side_effect=Exception("PDF extraction error"), create=True):
            with pytest.raises(IngestionError):
                service.ingest_patent("test.pdf", metadata)
