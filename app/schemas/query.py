from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class RetrievedChunk(BaseModel):
    content: str
    filename: str
    score: float


class VerificationResult(BaseModel):
    verdict: str
    reason: str


class QueryResponse(BaseModel):
    answer: str
    verification: VerificationResult
    retrieved_chunks: list[RetrievedChunk]