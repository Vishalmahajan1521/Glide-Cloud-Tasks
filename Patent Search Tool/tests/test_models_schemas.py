"""
Tests for Pydantic schemas
"""
import pytest
from pydantic import ValidationError
from app.models.schemas.search import SearchRequest, SearchFilters, SearchResult
from app.models.schemas.ingest import PatentMetadata


class TestSearchFilters:
    """Tests for SearchFilters schema"""
    
    def test_search_filters_empty(self):
        """Test SearchFilters with no filters"""
        filters = SearchFilters()
        assert filters.jurisdiction is None
        assert filters.assignee is None
        assert filters.filing_year_from is None
        assert filters.filing_year_to is None
        assert filters.patent_class is None
        assert filters.topic is None
    
    def test_search_filters_with_values(self):
        """Test SearchFilters with all values"""
        filters = SearchFilters(
            jurisdiction=["US", "EP"],
            assignee=["Company A", "Company B"],
            filing_year_from=2015,
            filing_year_to=2020,
            patent_class=["H01M"],
            topic="thermal_management"
        )
        
        assert filters.jurisdiction == ["US", "EP"]
        assert filters.assignee == ["Company A", "Company B"]
        assert filters.filing_year_from == 2015
        assert filters.filing_year_to == 2020
        assert filters.patent_class == ["H01M"]
        assert filters.topic == "thermal_management"
    
    def test_search_filters_partial(self):
        """Test SearchFilters with partial values"""
        filters = SearchFilters(
            jurisdiction=["US"],
            filing_year_from=2020
        )
        
        assert filters.jurisdiction == ["US"]
        assert filters.filing_year_from == 2020
        assert filters.filing_year_to is None


class TestSearchRequest:
    """Tests for SearchRequest schema"""
    
    def test_search_request_minimal(self):
        """Test SearchRequest with only required fields"""
        request = SearchRequest(query="battery technology")
        
        assert request.query == "battery technology"
        assert request.top_k == 20  # default value
        assert request.filters is None
    
    def test_search_request_with_filters(self):
        """Test SearchRequest with filters"""
        filters = SearchFilters(jurisdiction=["US"])
        request = SearchRequest(
            query="battery technology",
            top_k=10,
            filters=filters
        )
        
        assert request.query == "battery technology"
        assert request.top_k == 10
        assert request.filters.jurisdiction == ["US"]
    
    def test_search_request_empty_query(self):
        """Test SearchRequest with empty query (should still be valid)"""
        request = SearchRequest(query="")
        assert request.query == ""


class TestSearchResult:
    """Tests for SearchResult schema"""
    
    def test_search_result_creation(self):
        """Test creating a SearchResult"""
        result = SearchResult(
            patent_id="US12345678",
            score=0.95,
            title="Test Patent",
            assignee="Test Company",
            jurisdiction="US",
            matched_chunk_type="claim",
            explanation="High similarity with claim"
        )
        
        assert result.patent_id == "US12345678"
        assert result.score == 0.95
        assert result.title == "Test Patent"
        assert result.matched_chunk_type == "claim"
    
    def test_search_result_minimal(self):
        """Test SearchResult with only required fields"""
        result = SearchResult(
            patent_id="US12345678",
            score=0.8,
            matched_chunk_type="abstract",
            explanation="Matched abstract"
        )
        
        assert result.patent_id == "US12345678"
        assert result.score == 0.8
        assert result.title is None
        assert result.assignee is None


class TestPatentMetadata:
    """Tests for PatentMetadata schema"""
    
    def test_patent_metadata_creation(self):
        """Test creating PatentMetadata"""
        metadata = PatentMetadata(
            patent_id="US12345678",
            title="Test Patent",
            assignee="Test Company",
            jurisdiction="US",
            filing_year=2020,
            patent_class=["H01M", "B60L"]
        )
        
        assert metadata.patent_id == "US12345678"
        assert metadata.title == "Test Patent"
        assert metadata.assignee == "Test Company"
        assert metadata.jurisdiction == "US"
        assert metadata.filing_year == 2020
        assert metadata.patent_class == ["H01M", "B60L"]
    
    def test_patent_metadata_validation(self):
        """Test PatentMetadata validation"""
        # Should raise ValidationError for missing required fields
        with pytest.raises(ValidationError):
            PatentMetadata(
                patent_id="US12345678"
                # Missing required fields
            )
