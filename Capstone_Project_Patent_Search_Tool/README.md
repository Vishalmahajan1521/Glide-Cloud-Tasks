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
- **Intelligent Chunking**: Patents are segmented into chunks using sliding window technique for efficient vector search
- **Text Processing**: Currently processes Title and Abstract from patent data
- **RESTful API**: FastAPI backend with well-defined endpoints
- **Interactive UI**: Streamlit-based web interface for easy searching

## Uniqueness (Weights)

The system is designed with a weighted scoring framework for different patent sections, though currently all chunks are processed as "abstract" since the dataset (Patents.csv) contains only Title and Abstract columns:

- **Abstract**: Weight = 0.7
- **Claims**: Weight = 1.0
- **Description**: Weight = 0.4

**Note**: The weight system is implemented in the codebase and can be activated when ingesting full patent documents (PDFs or API data) that contain Claims and Description sections. Currently, when ingesting from CSV data, all chunks are marked as "abstract" type.

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
