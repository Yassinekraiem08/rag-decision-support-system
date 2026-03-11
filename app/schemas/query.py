from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class RetrievedChunk(BaseModel):
    content: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk]