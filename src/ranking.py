"""
ranking.py — Compute user interest similarity and rank articles
================================================================
Combines three signals to produce personalised article rankings:

  1. **Cosine similarity** between user-interest and article embeddings
     (Matthew — uses normalised dot product from sentence-transformers).
  2. **Recency boost** — exponential half-life decay so newer articles
     score higher than stale ones (Aaryan).
  3. **Source diversity rule** — greedy re-ranking that caps how many
     articles from any single source appear in the final digest (Aaryan).
"""

import math
import numpy as np
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. Cosine-similarity scoring  (Matthew)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Recency boost  (Aaryan)
# ---------------------------------------------------------------------------

def recency_boost(
    published_at: Optional[str],
    now: Optional[datetime] = None,
    half_life_days: float = 3.0,
) -> float:
    """
    Exponential-decay recency score in [0, 1].

    score = 2^(-age_days / half_life_days)
          = exp(-age_days * ln2 / half_life_days)

    Articles with no publish date get a neutral 0.5.
    """
    if published_at is None:
        return 0.5

    if now is None:
        now = datetime.now(timezone.utc)

    try:
        pub = datetime.fromisoformat(published_at)
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 0.5

    age_days = max((now - pub).total_seconds() / 86_400, 0.0)
    decay_constant = math.log(2) / half_life_days
    return math.exp(-decay_constant * age_days)


# ---------------------------------------------------------------------------
# 3. Source diversity rule  (Aaryan)
# ---------------------------------------------------------------------------

def enforce_source_diversity(
    ranked: list,
    max_per_source: int = 3,
    target_n: Optional[int] = None,
) -> list:
    """
    Greedy re-ranking that limits articles from any single source.

    Walks the pre-sorted list and accepts articles until a source hits
    ``max_per_source``.  Overflow articles are appended at the end so
    nothing is lost — they just rank lower.
    """
    source_counts: dict[str, int] = {}
    promoted: list = []
    deferred: list = []

    for item in ranked:
        row = item[0]
        key = (row["source"] or "unknown").lower().strip()
        count = source_counts.get(key, 0)
        if count < max_per_source:
            promoted.append(item)
            source_counts[key] = count + 1
        else:
            deferred.append(item)

    result = promoted + deferred
    if target_n is not None:
        result = result[:target_n]
    return result


# ---------------------------------------------------------------------------
# 4. Combined ranking pipeline
# ---------------------------------------------------------------------------

def rank_articles(
    conn,
    user_id,
    top_k=10,
    similarity_weight=0.6,
    recency_weight=0.4,
    half_life_days=3.0,
    max_per_source=3,
    now=None,
):
    """
    Full ranking pipeline: similarity → recency boost → diversity filter.
    Falls back to recency-only ranking when no user embedding exists.
    """
    user_vec = load_user_embedding(conn, user_id)
    articles = load_candidate_articles(conn)

    if user_vec is not None:
        scored = compute_scores(user_vec, articles)
    else:
        # Cold-start: all articles get similarity 0, ranked by recency only
        scored = [(row, 0.0) for row in articles
                  if row["embedding"] is not None]

    # Blend similarity + recency into a combined score
    blended = []
    for row, sim_score in scored:
        rec = recency_boost(row["published_at"], now=now, half_life_days=half_life_days)
        combined = similarity_weight * sim_score + recency_weight * rec
        blended.append((row, combined))

    # Sort by combined score
    blended.sort(key=lambda x: x[1], reverse=True)

    # Enforce source diversity
    diverse = enforce_source_diversity(
        blended, max_per_source=max_per_source, target_n=top_k,
    )

    return diverse


def format_results(ranked):
    return [
        {
            "id": row["id"],
            "content": row["content"], 
            "title": row["title"],
            "source": row["source"],
            "score": score,
        }
        for row, score in ranked
    ]


def get_top_articles_for_user(conn, user_id, top_k=10, **kwargs):
    ranked = rank_articles(conn, user_id, top_k=top_k, **kwargs)
    return format_results(ranked)
