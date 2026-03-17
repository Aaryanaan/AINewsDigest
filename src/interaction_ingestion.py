"""
interaction_ingestion.py — User engagement event capture + mock generator
=========================================================================
Stores events in the shared SQLite database (interactions table) so they
sit alongside Kevin's articles and the user profiles.

The mock generator can operate in two modes:
  1. "real" — pull actual article IDs from the articles table
  2. "synthetic" — generate fake IDs (fallback if articles haven't been ingested yet)
"""

from __future__ import annotations

import json
import uuid
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Event type enum
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    DIGEST_SENT = "digest_sent"
    DIGEST_OPENED = "digest_opened"
    ARTICLE_IMPRESSION = "article_impression"
    ARTICLE_CLICK = "article_click"
    ARTICLE_SAVE = "article_save"
    FEEDBACK_LIKE = "feedback_like"
    FEEDBACK_DISLIKE = "feedback_dislike"


# ---------------------------------------------------------------------------
# Event schema (Pydantic — application layer)
# ---------------------------------------------------------------------------

class InteractionEvent(BaseModel):
    """Single user-interaction event."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(description="FK → users.user_id")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: EventType
    article_id: Optional[str] = Field(
        default=None, description="FK → articles.id"
    )
    digest_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SQLite-backed interaction store
# ---------------------------------------------------------------------------

class InteractionStore:
    """
    Read/write interaction events against the shared SQLite database.
    Replaces the earlier JSONL approach so that everything (articles,
    users, interactions) lives in one queryable DB.
    """

    def __init__(self, conn: sqlite3.Connection):
        conn.row_factory = sqlite3.Row
        self.conn = conn

    # ── Write ─────────────────────────────────────────────────────────────

    def log_event(self, event: InteractionEvent) -> InteractionEvent:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO interactions
                (event_id, user_id, timestamp, event_type,
                 article_id, digest_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.user_id,
                event.timestamp.isoformat(),
                event.event_type.value,
                event.article_id,
                event.digest_id,
                json.dumps(event.metadata),
            ),
        )
        self.conn.commit()
        return event

    def log_events(self, events: list[InteractionEvent]) -> int:
        rows = [
            (
                e.event_id,
                e.user_id,
                e.timestamp.isoformat(),
                e.event_type.value,
                e.article_id,
                e.digest_id,
                json.dumps(e.metadata),
            )
            for e in events
        ]
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO interactions
                (event_id, user_id, timestamp, event_type,
                 article_id, digest_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    # ── Read / query ──────────────────────────────────────────────────────

    def get_events_for_user(
        self,
        user_id: str,
        event_types: Optional[list[EventType]] = None,
        since: Optional[datetime] = None,
    ) -> list[InteractionEvent]:
        query = "SELECT * FROM interactions WHERE user_id = ?"
        params: list = [user_id]

        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            query += f" AND event_type IN ({placeholders})"
            params.extend(et.value for et in event_types)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp"
        return [self._row_to_event(r) for r in self.conn.execute(query, params)]

    def get_article_events(self, article_id: str) -> list[InteractionEvent]:
        rows = self.conn.execute(
            "SELECT * FROM interactions WHERE article_id = ? ORDER BY timestamp",
            (article_id,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]

    # ── Analytics ─────────────────────────────────────────────────────────

    def compute_ctr(self, user_id: str, digest_id: Optional[str] = None) -> float:
        base = "WHERE user_id = ?"
        params: list = [user_id]
        if digest_id:
            base += " AND digest_id = ?"
            params.append(digest_id)

        impressions = self.conn.execute(
            f"SELECT COUNT(*) FROM interactions {base} AND event_type = 'article_impression'",
            params,
        ).fetchone()[0]
        clicks = self.conn.execute(
            f"SELECT COUNT(*) FROM interactions {base} AND event_type = 'article_click'",
            params,
        ).fetchone()[0]
        return clicks / impressions if impressions else 0.0

    def get_liked_article_ids(self, user_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT article_id FROM interactions "
            "WHERE user_id = ? AND event_type = 'feedback_like' AND article_id IS NOT NULL",
            (user_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_clicked_article_ids(self, user_id: str) -> list[str]:
        """Useful for building / updating the interest embedding."""
        rows = self.conn.execute(
            "SELECT DISTINCT article_id FROM interactions "
            "WHERE user_id = ? AND event_type = 'article_click' AND article_id IS NOT NULL",
            (user_id,),
        ).fetchall()
        return [r[0] for r in rows]

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> InteractionEvent:
        return InteractionEvent(
            event_id=row["event_id"],
            user_id=row["user_id"],
            timestamp=row["timestamp"],
            event_type=row["event_type"],
            article_id=row["article_id"],
            digest_id=row["digest_id"],
            metadata=json.loads(row["metadata"] or "{}"),
        )


# ---------------------------------------------------------------------------
# Mock data generator
# ---------------------------------------------------------------------------

DEVICES = ["desktop", "mobile", "tablet"]


def _get_real_article_ids(conn: sqlite3.Connection, limit: int = 200) -> list[str]:
    """Pull actual article IDs from Kevin's articles table."""
    rows = conn.execute(
        "SELECT id FROM articles ORDER BY ingested_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [r[0] for r in rows]


def _make_synthetic_article_ids(n: int = 200) -> list[str]:
    """Fallback when articles table is empty."""
    return [f"art-{uuid.uuid4().hex[:8]}" for _ in range(n)]


def generate_mock_events(
    conn: sqlite3.Connection,
    user_ids: list[str],
    n_digests_per_user: int = 3,
    articles_per_digest: int = 10,
    click_rate: float = 0.35,
    like_rate: float = 0.15,
    dislike_rate: float = 0.05,
    save_rate: float = 0.08,
    open_rate: float = 0.70,
    seed: int = 42,
) -> list[InteractionEvent]:
    """
    Generate mock interaction data.

    Tries to use real article IDs from the articles table first;
    falls back to synthetic IDs if none exist yet.
    """
    rng = random.Random(seed)

    real_ids = _get_real_article_ids(conn)
    if real_ids:
        all_articles = real_ids
        print(f"[mock] Using {len(real_ids)} real article IDs from DB")
    else:
        all_articles = _make_synthetic_article_ids()
        print("[mock] No articles in DB yet — using synthetic article IDs")

    events: list[InteractionEvent] = []
    base_time = datetime.now(timezone.utc) - timedelta(weeks=n_digests_per_user)

    for user_id in user_ids:
        for week in range(n_digests_per_user):
            digest_id = f"digest-{uuid.uuid4().hex[:8]}"
            digest_time = base_time + timedelta(weeks=week, hours=rng.randint(7, 10))

            events.append(InteractionEvent(
                user_id=user_id,
                timestamp=digest_time,
                event_type=EventType.DIGEST_SENT,
                digest_id=digest_id,
            ))

            if rng.random() < open_rate:
                open_time = digest_time + timedelta(minutes=rng.randint(5, 480))
                events.append(InteractionEvent(
                    user_id=user_id,
                    timestamp=open_time,
                    event_type=EventType.DIGEST_OPENED,
                    digest_id=digest_id,
                    metadata={"device": rng.choice(DEVICES)},
                ))

                k = min(articles_per_digest, len(all_articles))
                digest_articles = rng.sample(all_articles, k=k)

                for rank, art_id in enumerate(digest_articles, 1):
                    imp_time = open_time + timedelta(seconds=rank * rng.randint(1, 5))

                    events.append(InteractionEvent(
                        user_id=user_id,
                        timestamp=imp_time,
                        event_type=EventType.ARTICLE_IMPRESSION,
                        article_id=art_id,
                        digest_id=digest_id,
                        metadata={"rank_position": rank},
                    ))

                    if rng.random() < click_rate:
                        dwell = rng.randint(10, 300)
                        click_time = imp_time + timedelta(seconds=rng.randint(2, 30))
                        events.append(InteractionEvent(
                            user_id=user_id,
                            timestamp=click_time,
                            event_type=EventType.ARTICLE_CLICK,
                            article_id=art_id,
                            digest_id=digest_id,
                            metadata={
                                "rank_position": rank,
                                "dwell_time_seconds": dwell,
                                "device": rng.choice(DEVICES),
                            },
                        ))

                        if rng.random() < like_rate:
                            events.append(InteractionEvent(
                                user_id=user_id,
                                timestamp=click_time + timedelta(seconds=dwell),
                                event_type=EventType.FEEDBACK_LIKE,
                                article_id=art_id,
                                digest_id=digest_id,
                            ))
                        elif rng.random() < dislike_rate:
                            events.append(InteractionEvent(
                                user_id=user_id,
                                timestamp=click_time + timedelta(seconds=dwell),
                                event_type=EventType.FEEDBACK_DISLIKE,
                                article_id=art_id,
                                digest_id=digest_id,
                            ))

                        if rng.random() < save_rate:
                            events.append(InteractionEvent(
                                user_id=user_id,
                                timestamp=click_time + timedelta(seconds=dwell + 5),
                                event_type=EventType.ARTICLE_SAVE,
                                article_id=art_id,
                                digest_id=digest_id,
                            ))

    events.sort(key=lambda e: e.timestamp)
    return events


# ---------------------------------------------------------------------------
# CLI — seed users + mock interactions into the shared DB
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from db import get_connection, ensure_all_tables
    from user_schema import (
        UserProfile, UserPreferences, DigestFrequency,
        upsert_user, get_all_active_users,
    )

    conn = get_connection()
    ensure_all_tables(conn)

    # ── 1. Seed sample users ──────────────────────────────────────────────
    sample_users = [
        UserProfile(
            email="aaryan.nanekar@gmail.com",
            display_name="Aaryan",
            preferences=UserPreferences(
                preferred_sources=["techcrunch", "arxiv"],
                max_articles=10,
            ),
        ),
        UserProfile(
            email="kevin.yao0304@gmail.com",
            display_name="Kevin",
            preferences=UserPreferences(
                digest_frequency=DigestFrequency.DAILY,
                preferred_sources=["reuters", "wired"],
                max_articles=8,
            ),
        ),
        UserProfile(
            email="matthewdelpreto@gmail.com",
            display_name="Matthew",
            preferences=UserPreferences(
                preferred_sources=["bbc", "ars-technica"],
                max_articles=12,
            ),
        ),
    ]
    for u in sample_users:
        upsert_user(conn, u)
    print(f"Seeded {len(sample_users)} users into DB\n")

    # Reload from DB so we have the persisted user_ids
    users = get_all_active_users(conn)
    user_ids = [u.user_id for u in users]

    # ── 2. Generate & store mock interactions ─────────────────────────────
    events = generate_mock_events(
        conn, user_ids,
        n_digests_per_user=4,
        articles_per_digest=10,
    )
    store = InteractionStore(conn)
    written = store.log_events(events)
    print(f"Wrote {written} interaction events\n")

    # ── 3. Summary stats ─────────────────────────────────────────────────
    # Also show how many articles are in the DB (Kevin's data)
    article_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    print(f"Articles in DB: {article_count}")
    print(f"Users in DB:    {len(users)}")
    print(f"Events in DB:   {store.count()}\n")

    for u in users:
        ctr = store.compute_ctr(u.user_id)
        liked = store.get_liked_article_ids(u.user_id)
        clicked = store.get_clicked_article_ids(u.user_id)
        print(f"  {u.display_name:10s}  CTR={ctr:.0%}  clicks={len(clicked)}  likes={len(liked)}")

    conn.close()
