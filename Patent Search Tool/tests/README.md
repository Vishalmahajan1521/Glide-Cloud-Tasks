# Tests

This directory contains unit tests for the Patent Search Tool.

## Test Structure

- `test_models_domain.py` - Tests for domain models (Patent, PatentChunk)
- `test_models_schemas.py` - Tests for Pydantic schemas (SearchRequest, SearchFilters, etc.)
- `test_ml_chunking.py` - Tests for text chunking functions
- `test_ml_embeddings.py` - Tests for embedding model (with mocking)
- `test_retrieval_qdrant_store.py` - Tests for QdrantStore (with mocking)
- `test_services_search.py` - Tests for SearchService (with mocking)
- `test_services_ingest.py` - Tests for IngestService (with mocking)
- `conftest.py` - Shared pytest fixtures and configuration

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test file:
```bash
pytest tests/test_models_domain.py
```

### Run with verbose output:
```bash
pytest -v
```

### Run with coverage:
```bash
pytest --cov=app --cov-report=html
```

### Run specific test:
```bash
pytest tests/test_models_domain.py::TestPatent::test_patent_creation
```

## Test Coverage

The tests cover:
- ✅ Domain models and data structures
- ✅ Pydantic schemas and validation
- ✅ ML chunking functions (section splitting, sliding window)
- ✅ Embedding model (with mocked API calls)
- ✅ QdrantStore operations (with mocked Qdrant client)
- ✅ SearchService functionality
- ✅ IngestService functionality

## Notes

- External dependencies (Qdrant, Ollama API) are mocked to avoid requiring running services
- Tests use pytest fixtures for common test data
- All tests are designed to be fast and isolated
