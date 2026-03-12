from concurrent.futures import ThreadPoolExecutor
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


def search_chunks_in_db(query: str, top_k: int = 4):
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

        return results[:top_k]
    finally:
        db.close()