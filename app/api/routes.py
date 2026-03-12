from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.query import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    VerificationResult,
)
from app.schemas.ingest import IngestResponse
from app.services.pgvector_store import search_chunks_in_db, store_document_chunks
from app.services.reranker import rerank_chunks
from app.services.generation import generate_answer, stream_answer
from app.services.verifier import verify_answer
from app.services.chunking import chunk_text
from app.utils.loaders import load_uploaded_text_file, load_uploaded_pdf_file

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    if file.filename.endswith(".txt"):
        text = await load_uploaded_text_file(file)
    elif file.filename.endswith(".pdf"):
        text = await load_uploaded_pdf_file(file)
    else:
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .pdf files are supported for now."
        )

    chunks = chunk_text(text, chunk_size=200, overlap=40)
    store_document_chunks(file.filename, chunks)

    return IngestResponse(
        filename=file.filename,
        chunks_stored=len(chunks)
    )


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    retrieved = search_chunks_in_db(request.question, top_k=6)
    reranked = rerank_chunks(request.question, retrieved, top_k=3)

    answer = generate_answer(request.question, reranked)
    verification = verify_answer(request.question, answer, reranked)

    retrieved_chunks = [
        RetrievedChunk(content=chunk, filename=filename, score=score)
        for chunk, filename, score in reranked
    ]

    return QueryResponse(
        answer=answer,
        verification=VerificationResult(
            verdict=verification["verdict"],
            reason=verification["reason"],
        ),
        retrieved_chunks=retrieved_chunks,
    )


@router.post("/query/stream")
def query_rag_stream(request: QueryRequest):
    retrieved = search_chunks_in_db(request.question, top_k=6)
    reranked = rerank_chunks(request.question, retrieved, top_k=3)

    return StreamingResponse(
        stream_answer(request.question, reranked),
        media_type="text/plain"
    )