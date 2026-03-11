from fastapi import APIRouter
from app.schemas.query import QueryRequest, QueryResponse, RetrievedChunk
from app.services.pgvector_store import search_chunks_in_db
from app.services.generation import generate_answer

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    retrieved = search_chunks_in_db(request.question, top_k=2)
    answer = generate_answer(request.question, retrieved)

    retrieved_chunks = [
        RetrievedChunk(content=chunk, score=score)
        for chunk, score in retrieved
    ]

    return QueryResponse(answer=answer, retrieved_chunks=retrieved_chunks)