# Capstone Project: Patent Search Tool

## What is this Project?

The Patent Search Tool is an intelligent semantic search system that enables users to search through patent documents using natural language queries. Instead of traditional keyword-based search, this tool uses vector embeddings and Retrieval Augmented Generation (RAG) to understand the semantic meaning of queries and find the most relevant patents.

## Tech Stack

- **Backend API**: FastAPI
- **Frontend UI**: Streamlit
- **Vector Database**: Qdrant
- **Embeddings**: Ollama (nomic-embed-text model)
- **Language**: Python 3.x
- **Validation**: Pydantic
- **Testing**: Pytest

## Features

- **Semantic Search**: Search patents using natural language queries instead of exact keywords
- **Advanced Filtering**: Filter results by:
  - Jurisdiction (US, EP, DE, WO)
  - Assignee
  - Filing Year Range
  - Patent Class
  - Topic
- **Intelligent Chunking**: Patents are segmented into different sections (claims, abstract, description) for more precise matching
- **Weighted Search Results**: Different sections are weighted differently based on their importance
- **RESTful API**: FastAPI backend with well-defined endpoints
- **Interactive UI**: Streamlit-based web interface for easy searching

## Uniqueness (Weights)

The system uses weighted scoring for different patent sections to prioritize more important content:

- **Claims**: Weight = 1.0 (highest priority - most legally significant)
- **Abstract**: Weight = 0.7 (high priority - summary of invention)
- **Description**: Weight = 0.4 (standard priority - detailed description)

This weighting ensures that matches in patent claims are ranked higher than matches in descriptions, providing more relevant and legally significant results.

## How to Run

### Prerequisites

1. Install Python 3.8 or higher
2. Install Ollama and pull the embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```
3. Set up Qdrant (local or cloud instance)

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd Capstone_Project_Patent_Search_Tool
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the project root with the following variables:
   ```
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   OLLAMA_URL=http://localhost:11434
   OLLAMA_MODEL=nomic-embed-text
   ```

### Running the Application

1. **Start the FastAPI Backend**:
   ```bash
   uvicorn app.main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`

2. **Start the Streamlit Frontend** (optional):
   ```bash
   streamlit run streamlit_app.py
   ```
   The UI will be available at `http://localhost:8501`

3. **Ingest Patent Data**:
   Use the batch ingest script or API endpoint to load patent data into the vector database:
   ```bash
   python scripts/batch_ingest.py
   ```

### API Endpoints

- `GET /api/v1/health` - Health check endpoint
- `POST /api/v1/ingest` - Ingest patent documents
- `POST /api/v1/search` - Search patents with semantic queries

### Testing

Run the test suite:
```bash
pytest tests/
```
