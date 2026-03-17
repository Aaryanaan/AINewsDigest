"""Embed articles and store normalized vectors with metadata."""

import sqlite3
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.db import ensure_all_tables

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "articles.db"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_batch(conn: sqlite3.Connection, limit: int) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, content FROM articles
        WHERE content IS NOT NULL AND (
            embedding IS NULL OR embedding_dim IS NULL OR embedding_model IS NULL
        )
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(norm):
        return vec
    return vec / norm


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    conn = get_connection()
    ensure_all_tables(conn)

    total = 0
    while True:
        rows = fetch_batch(conn, BATCH_SIZE)
        if not rows:
            break

        texts = [r["content"] for r in rows]
        ids = [r["id"] for r in rows]

        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)

        updates = []
        for art_id, emb in zip(ids, embeddings):
            if emb is None:
                continue
            emb = normalize(emb.astype(np.float32))
            updates.append(
                (
                    emb.tobytes(),
                    emb.shape[0],
                    MODEL_NAME,
                    art_id,
                )
            )

        conn.executemany(
            """
            UPDATE articles
            SET embedding = ?, embedding_dim = ?, embedding_model = ?
            WHERE id = ?
            """,
            updates,
        )
        conn.commit()
        total += len(updates)
        print(f"[embed] wrote {len(updates)} embeddings (total {total})")

    print(f"[embed] done. total updated: {total}")


if __name__ == "__main__":
    main()