# AI Decision Support System (Retrieval-Augmented Generation)

A Retrieval-Augmented Generation (RAG) system for document ingestion, semantic search, and citation-grounded AI responses. This project uses Python, FastAPI, and vector search with PostgreSQL + pgvector (with optional Pinecone support) to enable scalable decision support over unstructured documents.

## Features

- Document ingestion pipeline for PDF, DOCX, and text files
- Automatic text extraction and chunking
- Embedding generation for semantic search
- Vector similarity search using PostgreSQL + pgvector
- Citation-grounded LLM responses
- RESTful API built with FastAPI
- Modular backend design for future extension to Pinecone or other vector databases

## Tech Stack

- Python
- FastAPI
- PostgreSQL
- pgvector
- SQLAlchemy
- OpenAI API
- Docker Compose
- PyPDF / python-docx

## Project Structure

```text
ai-decision-support-system/
├── app/
│   ├── api/
│   │   └── routes/
│   ├── core/
│   ├── db/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── data/
├── tests/
├── .env
├── requirements.txt
├── docker-compose.yml
└── README.md
