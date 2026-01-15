"""
Tests for domain models
"""
import pytest
from app.models.domain.patent import Patent, PatentChunk


class TestPatent:
    """Tests for Patent dataclass"""
    
    def test_patent_creation(self):
        """Test creating a Patent instance"""
        patent = Patent(
            patent_id="US12345678",
            title="Test Patent",
            assignee="Test Company",
            jurisdiction="US",
            filing_year=2020,
            patent_class=["H01M", "B60L"]
        )
        
        assert patent.patent_id == "US12345678"
        assert patent.title == "Test Patent"
        assert patent.assignee == "Test Company"
        assert patent.jurisdiction == "US"
        assert patent.filing_year == 2020
        assert patent.patent_class == ["H01M", "B60L"]
    
    def test_patent_with_empty_classes(self):
        """Test Patent with empty patent_class list"""
        patent = Patent(
            patent_id="US12345678",
            title="Test Patent",
            assignee="Test Company",
            jurisdiction="US",
            filing_year=2020,
            patent_class=[]
        )
        
        assert patent.patent_class == []


class TestPatentChunk:
    """Tests for PatentChunk dataclass"""
    
    def test_patent_chunk_creation(self):
        """Test creating a PatentChunk instance"""
        chunk = PatentChunk(
            patent_id="US12345678",
            text="Sample text",
            chunk_type="abstract",
            section_priority=0.7
        )
        
        assert chunk.patent_id == "US12345678"
        assert chunk.text == "Sample text"
        assert chunk.chunk_type == "abstract"
        assert chunk.section_priority == 0.7
        assert chunk.claim_number is None
        assert chunk.chunk_index is None
    
    def test_patent_chunk_with_optional_fields(self):
        """Test PatentChunk with optional fields"""
        chunk = PatentChunk(
            patent_id="US12345678",
            text="Sample text",
            chunk_type="claim",
            section_priority=1.0,
            claim_number=1,
            chunk_index=0
        )
        
        assert chunk.claim_number == 1
        assert chunk.chunk_index == 0
