"""
Pytest configuration and shared fixtures
"""
import pytest
from unittest.mock import Mock, MagicMock
from typing import List


@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient for testing"""
    client = Mock()
    client.get_collections.return_value = Mock(collections=[])
    client.recreate_collection.return_value = None
    client.create_payload_index.return_value = None
    client.delete_collection.return_value = None
    client.upsert.return_value = None
    client.query_points.return_value = Mock(points=[])
    return client


@pytest.fixture
def mock_embedding_response():
    """Mock embedding response from Ollama"""
    return {
        "embedding": [0.1] * 768  # 768-dimensional vector
    }


@pytest.fixture
def sample_patent_metadata():
    """Sample patent metadata for testing"""
    return {
        "patent_id": "US12345678",
        "title": "Test Patent Title",
        "assignee": "Test Company",
        "jurisdiction": "US",
        "filing_year": 2020,
        "patent_class": ["H01M", "B60L"]
    }


@pytest.fixture
def sample_patent_text():
    """Sample patent text for testing"""
    return """
    Abstract
    This is a test abstract for a patent about battery technology.
    
    Description
    This is a detailed description of the invention. It includes various
    technical details about the battery thermal management system.
    
    Claims
    Claim 1: A battery system comprising a thermal management unit.
    Claim 2: The system of claim 1, further comprising a cooling mechanism.
    """


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing"""
    return [
        {
            "text": "This is a test abstract for a patent about battery technology.",
            "chunk_type": "abstract",
            "section_priority": 0.7,
            "chunk_index": 0
        },
        {
            "text": "This is a detailed description of the invention.",
            "chunk_type": "description",
            "section_priority": 0.4,
            "chunk_index": 0
        },
        {
            "text": "Claim 1: A battery system comprising a thermal management unit.",
            "chunk_type": "claim",
            "section_priority": 1.0,
            "claim_number": 1,
            "chunk_index": 0
        }
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing"""
    return [[0.1] * 768, [0.2] * 768, [0.3] * 768]


@pytest.fixture
def sample_search_results():
    """Sample search results from Qdrant"""
    from unittest.mock import Mock
    
    points = []
    for i in range(3):
        point = Mock()
        point.score = 0.9 - (i * 0.1)
        point.payload = {
            "text": f"Sample text {i}",
            "patent_id": f"US1234567{i}",
            "title": f"Test Patent {i}",
            "assignee": "Test Company",
            "jurisdiction": "US",
            "filing_year": 2020,
            "patent_class": ["H01M"],
            "chunk_type": "claim" if i == 0 else "abstract"
        }
        points.append(point)
    
    result = Mock()
    result.points = points
    return result
