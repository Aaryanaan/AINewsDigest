"""Compute user interest embeddings as weighted averages of article embeddings."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

from src.db import get_connection, ensure_all_tables

MODEL_NAME = "all-MiniLM-L6-v2"
LOOKBACK_DAYS = 90
MIN_INTERACTIONS = 3
WEIGHTS: Dict[str, float] = {
    "article_impression": 0.0,
    "article_click": 1.0,
    "article_save": 2.0,
    "feedback_like": 2.0,
    "feedback_dislike": -1.0,
}


def load_users(conn) -> List[Dict]:
    return conn.execute(
        "SELECT user_id FROM users WHERE is_active = 1"
    ).fetchall()


def load_interactions(conn, user_id: str, since_iso: str) -> List[Dict]:
    placeholders = ",".join("?" for _ in WEIGHTS.keys())
    query = (
        "SELECT article_id, event_type FROM interactions "
        "WHERE user_id = ? AND timestamp >= ? "
        f"AND event_type IN ({placeholders})"
    )
    params = [user_id, since_iso, *WEIGHTS.keys()]
    return conn.execute(query, params).fetchall()


def load_article_embeddings(conn, article_ids: List[str]) -> Dict[str, np.ndarray]:
    if not article_ids:
        return {}
    placeholders = ",".join("?" for _ in article_ids)
    rows = conn.execute(
        f"SELECT id, embedding, embedding_dim FROM articles WHERE id IN ({placeholders})",
        article_ids,
    ).fetchall()
    out: Dict[str, np.ndarray] = {}
    for row in rows:
        if row["embedding"] is None or row["embedding_dim"] is None:
            continue
        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        if vec.shape[0] != row["embedding_dim"]:
            continue
        out[row["id"]] = vec
    return out


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(norm):
        return vec
    return vec / norm


def compute_interest(embeddings: List[np.ndarray], weights: List[float]) -> Optional[np.ndarray]:
    if not embeddings:
        return None
    weighted = np.zeros_like(embeddings[0], dtype=np.float32)
    total = 0.0
    for emb, w in zip(embeddings, weights):
        if emb is None:
            continue
        weighted += emb * w
        total += abs(w)
    if total == 0:
        return None
    weighted /= total
    weighted = normalize(weighted)
    if np.linalg.norm(weighted) == 0:
        return None
    return weighted.astype(np.float32)


def update_user_interest(conn, user_id: str, vector: Optional[np.ndarray]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    if vector is None:
        conn.execute(
            """
            UPDATE users SET
                interest_embedding = NULL,
                interest_embedding_dim = NULL,
                interest_embedding_model = NULL,
                last_profile_update_at = ?
            WHERE user_id = ?
            """,
            (now, user_id),
        )
    else:
        conn.execute(
            """
            UPDATE users SET
                interest_embedding = ?,
                interest_embedding_dim = ?,
                interest_embedding_model = ?,
                last_profile_update_at = ?
            WHERE user_id = ?
            """,
            (vector.tobytes(), vector.shape[0], MODEL_NAME, now, user_id),
        )
    conn.commit()


def main() -> None:
    conn = get_connection()
    ensure_all_tables(conn)

    since = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    since_iso = since.isoformat()

    users = load_users(conn)
    print(f"[interest] active users: {len(users)}")

    updated = 0
    skipped = 0

    for user_row in users:
        user_id = user_row["user_id"]
        events = load_interactions(conn, user_id, since_iso)
        weighted_embs: List[np.ndarray] = []
        weights: List[float] = []

        # Collect article IDs with non-zero weights
        article_ids = [e["article_id"] for e in events if e["article_id"]]
        if not article_ids:
            skipped += 1
            update_user_interest(conn, user_id, None)
            continue

        embeddings_map = load_article_embeddings(conn, article_ids)

        for event in events:
            aid = event["article_id"]
            weight = WEIGHTS.get(event["event_type"], 0.0)
            if weight == 0.0:
                continue
            emb = embeddings_map.get(aid)
            if emb is None:
                continue
            emb = normalize(emb.astype(np.float32))
            weighted_embs.append(emb)
            weights.append(weight)

        if len(weighted_embs) < MIN_INTERACTIONS:
            skipped += 1
            update_user_interest(conn, user_id, None)
            continue

        vector = compute_interest(weighted_embs, weights)
        update_user_interest(conn, user_id, vector)
        if vector is not None:
            updated += 1

    print(f"[interest] updated {updated} users, skipped {skipped}")


if __name__ == "__main__":
    main()
