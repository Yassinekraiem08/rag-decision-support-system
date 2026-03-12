# AI Decision Support System (RAG)

A Retrieval-Augmented Generation (RAG) based AI decision support system built with **Python**, **FastAPI**, **PostgreSQL + pgvector**, and **OpenAI APIs**. The system ingests PDF/TXT documents, splits them into chunks, generates embeddings, performs native vector similarity search, and serves citation-grounded responses through REST endpoints.

## Features

- Batch ingestion of `.txt` and `.pdf` documents
- Document upload through API
- Text chunking with overlap
- Embedding generation using OpenAI
- Vector storage in PostgreSQL with pgvector
- Native vector similarity search inside PostgreSQL
- Citation-grounded answer generation with filename-based citations
- Standard and streaming query endpoints
- FastAPI Swagger documentation

## Tech Stack

- Python
- FastAPI
- PostgreSQL
- pgvector
- SQLAlchemy
- OpenAI API
- Docker
- PyPDF

## Architecture

### Ingestion Pipeline
1. Upload or batch-load documents
2. Extract raw text from TXT/PDF files
3. Split text into chunks
4. Generate embeddings for chunks
5. Store document metadata, chunk text, and vectors in PostgreSQL

### Query Pipeline
1. Submit a question through `/query` or `/query/stream`
2. Generate query embedding
3. Perform native pgvector similarity search in PostgreSQL
4. Retrieve the top matching chunks
5. Generate a grounded answer using retrieved context
6. Return the answer with filename-grounded citations

## API Endpoints

### `POST /ingest`
Uploads a `.txt` or `.pdf` file, extracts text, chunks it, embeds it, and stores it in PostgreSQL.

Example response:

```json
{
  "filename": "sample_doc.txt",
  "chunks_stored": 7
}
POST /query

Returns a grounded answer and retrieved chunks.

Example request:

{
  "question": "What is embodied AI?"
}

Example response:

{
  "answer": "Embodied AI refers to AI systems that interact with and learn from physical environments [qya.pdf].",
  "retrieved_chunks": [
    {
      "content": "Embodied AI refers to...",
      "filename": "qya.pdf",
      "score": 0.71
    }
  ]
}
POST /query/stream

Streams the generated response progressively.

Batch Ingestion

The project also supports batch indexing from a folder of documents.

Example structure:

docs/
├── arxiv/
│   ├── paper1.pdf
│   ├── paper2.pdf
├── gutenberg/
│   ├── book1.txt
│   ├── book2.txt
└── wiki/
    ├── ai.txt
    ├── robotics.txt

Run:

python ingest_folder.py

Example output:

Done. Indexed 18 documents and 1673 chunks.
Project Structure
app/
├── api/
├── core/
├── models/
├── schemas/
├── services/
└── utils/

docs/
docker-compose.yml
init_db.py
ingest_folder.py
requirements.txt
Setup
1. Clone the repository
git clone git@github.com:Yassinekraiem08/rag-decision-support-system.git
cd rag-decision-support-system
2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables

```

## Architecture

This project implements a Retrieval-Augmented Generation (RAG) pipeline combining
semantic vector retrieval and keyword ranking.

See system architecture below.

![RAG Architecture](./rag_architecture.png)

## Create a .env file:

OPENAI_API_KEY=your_key_here
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/ragdb
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4.1-mini
5. Start PostgreSQL with pgvector
docker compose up -d
6. Initialize the database
python init_db.py
7. Run the API
uvicorn app.main:app --reload

Open:

http://127.0.0.1:8000/docs
Why This Project Matters

Traditional LLMs can hallucinate or answer without evidence. This project improves reliability by retrieving relevant external context before generation, enabling more grounded, traceable, and source-aware responses.
