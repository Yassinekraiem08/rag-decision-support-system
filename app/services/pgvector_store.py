from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import select, delete
from app.core.database import SessionLocal
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embeddings import generate_embedding, generate_embeddings
import logging
import os

logger = logging.getLogger(__name__)


def batched(items, batch_size=100):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def clear_database():
    db = SessionLocal()
    try:
        db.execute(delete(Chunk))
        db.execute(delete(Document))
        db.commit()
    finally:
        db.close()


def store_document_chunks(filename: str, chunks: list[str]):
    db = SessionLocal()

    try:
        existing = db.query(Document).filter_by(filename=filename).first()

        if existing:
            db.query(Chunk).filter_by(document_id=existing.id).delete()
            db.delete(existing)
            db.commit()

        document = Document(filename=filename)
        db.add(document)
        db.commit()
        db.refresh(document)

        chunk_batches = list(batched(chunks, batch_size=100))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(generate_embeddings, batch) for batch in chunk_batches]

            for chunk_batch, future in zip(chunk_batches, futures):
                embeddings = future.result()

                for chunk, embedding in zip(chunk_batch, embeddings):
                    db_chunk = Chunk(
                        document_id=document.id,
                        content=chunk,
                        embedding=embedding
                    )
                    db.add(db_chunk)

        db.commit()
    finally:
        db.close()


def search_chunks_in_db(query: str, top_k: int = 4, min_score: float = None):
    """
    Search for relevant chunks in the database with optional score filtering.

    Args:
        query: The search query
        top_k: Number of top results to return
        min_score: Minimum score threshold (default from env or 0.5)

    Returns:
        List of tuples (content, filename, score)
    """
    if min_score is None:
        min_score = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.5"))

    db = SessionLocal()

    try:
        query_embedding = generate_embedding(query)

        stmt = (
            select(
                Chunk.content,
                Document.filename,
                Chunk.embedding.cosine_distance(query_embedding).label("distance")
            )
            .join(Document, Chunk.document_id == Document.id)
            .order_by(Chunk.embedding.cosine_distance(query_embedding))
            .limit(10)
        )

        rows = db.execute(stmt).all()

        keywords = query.lower().split()
        results = []

        for content, filename, distance in rows:
            vector_score = 1 - float(distance)

            keyword_score = sum(
                content.lower().count(word)
                for word in keywords
            )

            final_score = vector_score + (0.1 * keyword_score)

            results.append((content, filename, final_score))

        results.sort(key=lambda x: x[2], reverse=True)

        # Apply score threshold filtering
        filtered = [r for r in results if r[2] >= min_score]

        # Log if chunks were filtered
        if len(filtered) < len(results):
            logger.info(f"Filtered {len(results) - len(filtered)} chunks below min_score={min_score}")

        # Handle edge case: if too few chunks pass threshold, return what we have
        if len(filtered) < top_k:
            logger.warning(f"Only {len(filtered)}/{top_k} chunks passed min_score={min_score}. Returning available chunks.")
            return filtered

        return filtered[:top_k]
    finally:
        db.close()