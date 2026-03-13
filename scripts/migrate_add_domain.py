#!/usr/bin/env python3
"""
Migration: Add domain column to documents table.

Run once to update an existing database that predates the domain tagging feature.
Safe to re-run — uses IF NOT EXISTS and only updates NULL rows.

Usage:
    python scripts/migrate_add_domain.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import text
from app.core.database import engine


def run():
    with engine.connect() as conn:
        # Add column if it doesn't exist yet
        conn.execute(text(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS domain VARCHAR NOT NULL DEFAULT 'technical'"
        ))

        # Classify existing Gutenberg files as literary
        result = conn.execute(text(
            "UPDATE documents SET domain = 'literary' WHERE filename ~ '^pg[0-9]+\\.txt$'"
        ))
        literary_updated = result.rowcount

        # Ensure any remaining NULLs are set (shouldn't occur due to DEFAULT, but be safe)
        conn.execute(text(
            "UPDATE documents SET domain = 'technical' WHERE domain IS NULL"
        ))

        conn.commit()

    print(f"Migration complete.")
    print(f"  - domain column added (or already existed)")
    print(f"  - {literary_updated} Gutenberg documents tagged as 'literary'")
    print(f"  - All other documents tagged as 'technical'")


if __name__ == "__main__":
    run()
