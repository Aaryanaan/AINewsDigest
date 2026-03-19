"""
ranking.py — Compute user interest similarity and rank articles

This module calculates cosine similarity between user interest embeddings
and article embeddings to generate personalized rankings. It uses
NumPy arrays for vector operations and assumes embeddings are already
stored in the SQLite database.

"""
import numpy as np
from datetime import datetime

def load_user_embedding(conn, user_id):
    row = conn.execute(
        "SELECT interest_embedding, interest_embedding_dim FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    if not row or row["interest_embedding"] is None:
        return None

    vec = np.frombuffer(row["interest_embedding"], dtype=np.float32)
    if vec.shape[0] != row["interest_embedding_dim"]:
        return None

    return vec

def load_candidate_articles(conn, limit=500):
    return conn.execute(
        """
        SELECT id, title, content, embedding, embedding_dim, source, published_at
        FROM articles
        WHERE embedding IS NOT NULL
        ORDER BY published_at DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()

def compute_scores(user_vec, article_rows):
    results = []

    for row in article_rows:
        emb = row["embedding"]
        dim = row["embedding_dim"]

        if emb is None or dim is None:
            continue

        article_vec = np.frombuffer(emb, dtype=np.float32)
        if article_vec.shape[0] != dim:
            continue

        # cosine similarity (since normalized)
        score = float(np.dot(user_vec, article_vec))

        results.append((row, score))

    return results

def rank_articles(conn, user_id, top_k=10):
    user_vec = load_user_embedding(conn, user_id)
    if user_vec is None:
        return []

    articles = load_candidate_articles(conn)

    scored = compute_scores(user_vec, articles)

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)

    return ranked[:top_k]

def format_results(ranked):
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "source": row["source"],
            "score": score,
        }
        for row, score in ranked
    ]

def get_top_articles_for_user(conn, user_id, top_k=10):
    ranked = rank_articles(conn, user_id, top_k)
    return format_results(ranked)

def apply_time_decay(score, published_at):
    if not published_at:
        return score

    age_days = (datetime.utcnow() - datetime.fromisoformat(published_at)).days
    decay = 1 / (1 + 0.1 * age_days)

    return score * decay