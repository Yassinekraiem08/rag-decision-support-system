from sqlalchemy import select, delete
from app.core.database import SessionLocal
from app.models.chunk import Chunk
from app.services.embeddings import generate_embedding


def clear_chunks_table():
    db = SessionLocal()
    try:
        db.execute(delete(Chunk))
        db.commit()
    finally:
        db.close()


def store_chunks_in_db(chunks: list[str]):
    db = SessionLocal()

    try:
        for chunk in chunks:
            embedding = generate_embedding(chunk)

            db_chunk = Chunk(
                content=chunk,
                embedding=embedding
            )
            db.add(db_chunk)

        db.commit()
    finally:
        db.close()


def search_chunks_in_db(query: str, top_k: int = 3):
    db = SessionLocal()

    try:
        query_embedding = generate_embedding(query)

        stmt = (
            select(
                Chunk.content,
                Chunk.embedding.cosine_distance(query_embedding).label("distance")
            )
            .order_by(Chunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )

        rows = db.execute(stmt).all()

        results = []
        for content, distance in rows:
            score = 1 - float(distance)
            results.append((content, score))

        return results
    finally:
        db.close()