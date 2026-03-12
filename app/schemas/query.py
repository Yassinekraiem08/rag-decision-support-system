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
    references: list[str]
    confidence: float
    confidence_reasoning: str
    verification: VerificationResult
    retrieved_chunks: list[RetrievedChunk]