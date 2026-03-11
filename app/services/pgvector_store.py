from sqlalchemy import select, delete
from app.core.database import SessionLocal
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embeddings import generate_embedding, generate_embeddings


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
        document = Document(filename=filename)
        db.add(document)
        db.commit()
        db.refresh(document)

        for chunk_batch in batched(chunks, batch_size=100):
            embeddings = generate_embeddings(chunk_batch)

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


def search_chunks_in_db(query: str, top_k: int = 3):
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
            .limit(top_k)
        )

        rows = db.execute(stmt).all()

        results = []
        for content, filename, distance in rows:
            score = 1 - float(distance)
            results.append((content, filename, score))

        return results
    finally:
        db.close()