from sqlalchemy import Column, Integer, String
from app.core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    domain = Column(String, nullable=False, server_default="technical")