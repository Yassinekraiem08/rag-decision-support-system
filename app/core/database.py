from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_size=5,           # number of persistent connections
    max_overflow=10,       # extra connections allowed under load
    pool_timeout=30,       # seconds to wait for a connection before error
    pool_recycle=1800,     # recycle connections every 30 min (avoids stale conn errors)
    pool_pre_ping=True,    # verify connection is alive before using it
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()