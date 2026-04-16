"""Database helpers for SQLite-backed ingestion, users, and interactions."""

import sqlite3
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "articles.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def ensure_all_tables(conn: sqlite3.Connection) -> None:
    # Articles table (matches ingest.py schema) with embedding metadata columns
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            title TEXT,
            url TEXT NOT NULL,
            source TEXT,
            published_at TEXT,
            summary TEXT,
            content TEXT,
            topic TEXT,
            embedding BLOB,
            ingested_at TEXT NOT NULL,
            author TEXT,
            image_url TEXT,
            categories TEXT,
            source_feed_url TEXT,
            fetched_at TEXT,
            word_count INTEGER,
            canonical_url TEXT,
            guid TEXT,
            language TEXT,
            embedding_dim INTEGER,
            embedding_model TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
        """
    )
    _ensure_column(conn, "articles", "embedding_dim", "INTEGER")
    _ensure_column(conn, "articles", "embedding_model", "TEXT")

    # Users table with interest embedding stored as BLOB + metadata
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            display_name TEXT,
            digest_frequency TEXT,
            digest_day TEXT,
            digest_time TEXT,
            timezone TEXT,
            max_articles INTEGER,
            preferred_sources TEXT,
            blocked_sources TEXT,
            language TEXT,
            reading_level TEXT,
            interest_embedding BLOB,
            interest_embedding_dim INTEGER,
            interest_embedding_model TEXT,
            interest_topics TEXT,
            last_profile_update_at TEXT,
            is_active INTEGER,
            created_at TEXT,
            updated_at TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);
        """
    )

    # Interactions table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            event_id TEXT PRIMARY KEY,
            user_id TEXT,
            timestamp TEXT,
            event_type TEXT,
            article_id TEXT,
            digest_id TEXT,
            metadata TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_interactions_article ON interactions(article_id);
        """
    )

    # Digest run table for timed personalized digest generation
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS digest_runs (
            digest_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            window_key TEXT NOT NULL,
            window_start TEXT,
            window_end TEXT,
            scheduled_for TEXT,
            status TEXT NOT NULL,
            article_ids TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_digest_runs_user_window
        ON digest_runs(user_id, window_key);
        """
    )

    # Per-user per-window summary cache
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS article_summaries (
            user_id TEXT NOT NULL,
            article_id TEXT NOT NULL,
            window_key TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            model_name TEXT,
            generated_at TEXT NOT NULL,
            expires_at TEXT,
            PRIMARY KEY (user_id, article_id, window_key)
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_article_summaries_window
        ON article_summaries(window_key);
        """
    )
    conn.commit()
