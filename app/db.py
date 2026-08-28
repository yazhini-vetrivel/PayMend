"""
app/db.py

SQLite engine + session setup. Call init_db() once on startup
(e.g. from main.py or scripts/run_recovery_batch.py) to create tables.
"""

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base

DB_PATH = os.getenv("DATABASE_PATH", "recovery_agent.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# check_same_thread=False needed because SQLite + a single file handle
# may be touched from webhook handler threads as well as scripts.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # set True while debugging to see generated SQL
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Create all tables if they don't already exist. Safe to call repeatedly."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session():
    """
    Usage:
        with get_session() as session:
            session.add(obj)
            session.commit()
    Automatically rolls back on exception and always closes the session.
    """
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    # run: python -m app.db
    init_db()
    print(f"Database initialized at {DB_PATH}")
