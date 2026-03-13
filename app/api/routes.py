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
from app.services.generation import generate_answer, stream_answer, format_answer_with_references
from app.services.verifier import verify_answer
from app.services.confidence import calculate_confidence, is_above_confidence_threshold
from app.services.chunking import chunk_text
from app.utils.loaders import load_uploaded_text_file, load_uploaded_pdf_file

router = APIRouter()


def classify_domain(filename: str) -> str:
    """Classify a document as 'literary' (Gutenberg) or 'technical' based on filename."""
    import re
    if re.match(r"pg\d+\.txt", filename):
        return "literary"
    return "technical"


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
    domain = classify_domain(file.filename)
    store_document_chunks(file.filename, chunks, domain=domain)

    return IngestResponse(
        filename=file.filename,
        chunks_stored=len(chunks)
    )


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    retrieved = search_chunks_in_db(request.question, top_k=6, domain_filter="technical")
    reranked = rerank_chunks(request.question, retrieved, top_k=3)

    # Confidence threshold check — refuse before generating if corpus has no relevant info
    if not is_above_confidence_threshold(request.question):
        return QueryResponse(
            answer="I don't have enough relevant information in the provided documents to answer this question confidently.",
            references=[],
            confidence=0.0,
            confidence_reasoning="Retrieval confidence below threshold — query likely out of corpus scope.",
            verification=VerificationResult(verdict="UNSUPPORTED", reason="No sufficiently relevant documents found."),
            retrieved_chunks=[],
        )

    answer = generate_answer(request.question, reranked)
    verification = verify_answer(request.question, answer, reranked)

    # Format answer with references
    formatted = format_answer_with_references(answer, reranked)

    # Calculate confidence score
    confidence_result = calculate_confidence(reranked, verification["verdict"])

    retrieved_chunks = [
        RetrievedChunk(content=chunk, filename=filename, score=score)
        for chunk, filename, score in reranked
    ]

    return QueryResponse(
        answer=formatted["answer"],
        references=formatted["references"],
        confidence=confidence_result["confidence"],
        confidence_reasoning=confidence_result["reasoning"],
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