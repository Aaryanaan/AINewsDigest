"""Rank articles for users via cosine similarity.

Uses precomputed article embeddings and per-user interest embeddings
(saved by pipeline.build_user_interests) to select the top N articles
per active user. Excludes any article the user already interacted
with. Intended to be called either directly or by the end-to-end
pipeline runner.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.db import ensure_all_tables, get_connection
from src.user_schema import UserProfile, get_all_active_users


@dataclass
class RankedArticle:
    article_id: str
    title: Optional[str]
    url: str
    source: Optional[str]
    published_at: Optional[str]
    score: float


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(norm):
        return vec
    return vec / norm


def _load_articles(conn: sqlite3.Connection) -> Tuple[List[str], np.ndarray, List[Dict]]:
    rows = conn.execute(
        """
        SELECT id, title, url, source, published_at, embedding, embedding_dim
        FROM articles
        WHERE embedding IS NOT NULL AND embedding_dim IS NOT NULL
        """
    ).fetchall()

    ids: List[str] = []
    vectors: List[np.ndarray] = []
    meta: List[Dict] = []

    for row in rows:
        vec = np.frombuffer(row[5], dtype=np.float32)
        if vec.shape[0] != row[6]:
            continue
        vec = _normalize(vec.astype(np.float32))
        ids.append(row[0])
        vectors.append(vec)
        meta.append(
            {
                "title": row[1],
                "url": row[2],
                "source": row[3],
                "published_at": row[4],
            }
        )

    if not vectors:
        return [], np.empty((0, 0), dtype=np.float32), []

    matrix = np.vstack(vectors)
    return ids, matrix, meta


def _load_seen_article_ids(conn: sqlite3.Connection, user_id: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT article_id FROM interactions
        WHERE user_id = ? AND article_id IS NOT NULL
        """,
        (user_id,),
    ).fetchall()
    return {r[0] for r in rows if r[0]}


def _rank_for_user(
    user: UserProfile,
    article_ids: List[str],
    article_vecs: np.ndarray,
    article_meta: List[Dict],
    seen_ids: Iterable[str],
) -> List[RankedArticle]:
    if not user.interests.interest_embedding:
        return []
    if article_vecs.size == 0:
        return []

    user_vec = np.array(user.interests.interest_embedding, dtype=np.float32)
    user_vec = _normalize(user_vec)

    if user_vec.shape[0] != article_vecs.shape[1]:
        return []

    seen = set(seen_ids)
    candidate_indices = [i for i, aid in enumerate(article_ids) if aid not in seen]
    if not candidate_indices:
        return []

    candidate_matrix = article_vecs[candidate_indices]
    scores = candidate_matrix @ user_vec

    top_k = min(user.preferences.max_articles, len(candidate_indices))
    best_idx = np.argsort(scores)[::-1][:top_k]

    ranked: List[RankedArticle] = []
    for idx in best_idx:
        original_idx = candidate_indices[int(idx)]
        score = float(scores[int(idx)])
        ranked.append(
            RankedArticle(
                article_id=article_ids[original_idx],
                title=article_meta[original_idx]["title"],
                url=article_meta[original_idx]["url"],
                source=article_meta[original_idx]["source"],
                published_at=article_meta[original_idx]["published_at"],
                score=score,
            )
        )
    return ranked


def rank_all_users(conn: Optional[sqlite3.Connection] = None) -> Dict[str, List[RankedArticle]]:
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    try:
        ensure_all_tables(conn)
        article_ids, article_vecs, article_meta = _load_articles(conn)

        users = get_all_active_users(conn)
        results: Dict[str, List[RankedArticle]] = {}

        for user in users:
            seen_ids = _load_seen_article_ids(conn, user.user_id)
            ranked = _rank_for_user(user, article_ids, article_vecs, article_meta, seen_ids)
            if ranked:
                results[user.user_id] = ranked
        return results
    finally:
        if close_conn:
            conn.close()


def main() -> None:
    conn = get_connection()
    results = rank_all_users(conn)

    if not results:
        print("[rank] no users ranked (missing interest embeddings or embeddings)")
        return

    for user_id, articles in results.items():
        print(f"[rank] user {user_id}: {len(articles)} articles")
        for art in articles:
            source = art.source or "?"
            print(f"  - {art.score:.3f} :: {source} :: {art.title or art.url}")


if __name__ == "__main__":
    main()
