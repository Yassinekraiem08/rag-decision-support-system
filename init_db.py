from sqlalchemy import text
from app.core.database import engine, Base
from app.models.chunk import Chunk

with engine.connect() as connection:
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    connection.commit()

Base.metadata.create_all(bind=engine)

print("Database initialized successfully.")