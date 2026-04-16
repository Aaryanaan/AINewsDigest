"""Digest orchestration for timed personalized summaries."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.db import ensure_all_tables
from src.interaction_ingestion import EventType, InteractionEvent, InteractionStore
from src.rank import rank_all_users
from src.summarize import summarize_article
from src.user_schema import DayOfWeek, DigestFrequency, UserProfile, get_all_active_users


DAY_INDEX = {
    DayOfWeek.MONDAY.value: 0,
    DayOfWeek.TUESDAY.value: 1,
    DayOfWeek.WEDNESDAY.value: 2,
    DayOfWeek.THURSDAY.value: 3,
    DayOfWeek.FRIDAY.value: 4,
    DayOfWeek.SATURDAY.value: 5,
    DayOfWeek.SUNDAY.value: 6,
}


@dataclass
class DigestContext:
    window_key: str
    window_start_utc: datetime
    window_end_utc: datetime
    scheduled_for_utc: datetime


@dataclass
class DigestArticle:
    article_id: str
    title: Optional[str]
    url: str
    source: Optional[str]
    published_at: Optional[str]
    score: float
    summary_text: Optional[str]


@dataclass
class DigestResult:
    digest_id: str
    user_id: str
    window_key: str
    status: str
    articles: List[DigestArticle]


def _parse_digest_time(value: str) -> time:
    parts = value.split(":")
    if len(parts) != 2:
        return time(hour=8, minute=0)
    try:
        hour = max(0, min(23, int(parts[0])))
        minute = max(0, min(59, int(parts[1])))
        return time(hour=hour, minute=minute)
    except ValueError:
        return time(hour=8, minute=0)


def _user_zone(user: UserProfile) -> ZoneInfo:
    try:
        return ZoneInfo(user.preferences.timezone)
    except Exception:
        return ZoneInfo("UTC")


def digest_context_for_user(user: UserProfile, now_utc: Optional[datetime] = None) -> DigestContext:
    now = now_utc or datetime.now(timezone.utc)
    local_now = now.astimezone(_user_zone(user))
    digest_clock = _parse_digest_time(user.preferences.digest_time)

    if user.preferences.digest_frequency == DigestFrequency.DAILY:
        local_date = local_now.date()
        start_local = datetime.combine(local_date, time.min, tzinfo=local_now.tzinfo)
        end_local = start_local + timedelta(days=1)
        scheduled_local = datetime.combine(local_date, digest_clock, tzinfo=local_now.tzinfo)
        window_key = f"daily:{local_date.isoformat()}"
    else:
        target_weekday = DAY_INDEX.get(user.preferences.digest_day.value, 0)
        start_of_week = local_now.date() - timedelta(days=local_now.weekday())
        start_local = datetime.combine(start_of_week, time.min, tzinfo=local_now.tzinfo)
        end_local = start_local + timedelta(days=7)
        scheduled_day = start_of_week + timedelta(days=target_weekday)
        scheduled_local = datetime.combine(scheduled_day, digest_clock, tzinfo=local_now.tzinfo)
        year, week, _ = scheduled_day.isocalendar()
        window_key = f"weekly:{year}-W{week:02d}"

    return DigestContext(
        window_key=window_key,
        window_start_utc=start_local.astimezone(timezone.utc),
        window_end_utc=end_local.astimezone(timezone.utc),
        scheduled_for_utc=scheduled_local.astimezone(timezone.utc),
    )


def is_digest_due(user: UserProfile, now_utc: Optional[datetime] = None) -> bool:
    now = now_utc or datetime.now(timezone.utc)
    ctx = digest_context_for_user(user=user, now_utc=now)
    return now >= ctx.scheduled_for_utc


def _get_digest_run(conn: sqlite3.Connection, user_id: str, window_key: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM digest_runs WHERE user_id = ? AND window_key = ?
        """,
        (user_id, window_key),
    ).fetchone()


def _create_or_update_digest_run(
    conn: sqlite3.Connection,
    *,
    digest_id: str,
    user_id: str,
    context: DigestContext,
    status: str,
    article_ids: List[str],
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO digest_runs (
            digest_id, user_id, window_key, window_start, window_end,
            scheduled_for, status, article_ids, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, window_key) DO UPDATE SET
            status=excluded.status,
            article_ids=excluded.article_ids,
            updated_at=excluded.updated_at
        """,
        (
            digest_id,
            user_id,
            context.window_key,
            context.window_start_utc.isoformat(),
            context.window_end_utc.isoformat(),
            context.scheduled_for_utc.isoformat(),
            status,
            json.dumps(article_ids),
            now_iso,
            now_iso,
        ),
    )


def _load_article_records(conn: sqlite3.Connection, article_ids: List[str]) -> Dict[str, sqlite3.Row]:
    if not article_ids:
        return {}
    placeholders = ",".join("?" for _ in article_ids)
    rows = conn.execute(
        f"""
        SELECT id, title, url, source, published_at, content, summary
        FROM articles
        WHERE id IN ({placeholders})
        """,
        article_ids,
    ).fetchall()
    return {row["id"]: row for row in rows}


def _get_cached_summary(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    article_id: str,
    window_key: str,
) -> Optional[str]:
    row = conn.execute(
        """
        SELECT summary_text, expires_at FROM article_summaries
        WHERE user_id = ? AND article_id = ? AND window_key = ?
        """,
        (user_id, article_id, window_key),
    ).fetchone()
    if not row:
        return None

    expires_at = row["expires_at"]
    if expires_at:
        try:
            if datetime.fromisoformat(expires_at) < datetime.now(timezone.utc):
                return None
        except Exception:
            return None
    return row["summary_text"]


def _upsert_summary(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    article_id: str,
    window_key: str,
    summary_text: str,
    model_name: str,
    expires_at: Optional[datetime],
) -> None:
    generated_at = datetime.now(timezone.utc)
    conn.execute(
        """
        INSERT INTO article_summaries (
            user_id, article_id, window_key, summary_text,
            model_name, generated_at, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, article_id, window_key) DO UPDATE SET
            summary_text=excluded.summary_text,
            model_name=excluded.model_name,
            generated_at=excluded.generated_at,
            expires_at=excluded.expires_at
        """,
        (
            user_id,
            article_id,
            window_key,
            summary_text,
            model_name,
            generated_at.isoformat(),
            expires_at.isoformat() if expires_at else None,
        ),
    )


def _summary_ttl_for_user(user: UserProfile) -> timedelta:
    if user.preferences.digest_frequency == DigestFrequency.DAILY:
        return timedelta(hours=24)
    return timedelta(days=7)


def build_digest_for_user(
    conn: sqlite3.Connection,
    *,
    user: UserProfile,
    force_refresh: bool = False,
    log_sent_event: bool = False,
) -> DigestResult:
    ensure_all_tables(conn)
    context = digest_context_for_user(user)

    ranked_by_user = rank_all_users(conn)
    ranked = ranked_by_user.get(user.user_id, [])
    article_ids = [item.article_id for item in ranked]

    existing_run = _get_digest_run(conn, user.user_id, context.window_key)
    digest_id = existing_run["digest_id"] if existing_run else f"digest-{uuid.uuid4().hex[:8]}"

    _create_or_update_digest_run(
        conn,
        digest_id=digest_id,
        user_id=user.user_id,
        context=context,
        status="ranked",
        article_ids=article_ids,
    )

    article_records = _load_article_records(conn, article_ids)
    ttl = _summary_ttl_for_user(user)
    digest_articles: List[DigestArticle] = []

    for ranked_item in ranked:
        record = article_records.get(ranked_item.article_id)
        if not record:
            continue

        cached_summary = None if force_refresh else _get_cached_summary(
            conn,
            user_id=user.user_id,
            article_id=ranked_item.article_id,
            window_key=context.window_key,
        )

        if cached_summary:
            summary_text = cached_summary
        else:
            raw_content = record["content"] or record["summary"] or ranked_item.title or ""
            summary_text, model_name = summarize_article(
                title=record["title"] or ranked_item.title or "Untitled",
                source=record["source"],
                content=raw_content,
            )
            _upsert_summary(
                conn,
                user_id=user.user_id,
                article_id=ranked_item.article_id,
                window_key=context.window_key,
                summary_text=summary_text,
                model_name=model_name,
                expires_at=datetime.now(timezone.utc) + ttl,
            )

        digest_articles.append(
            DigestArticle(
                article_id=ranked_item.article_id,
                title=record["title"],
                url=record["url"],
                source=record["source"],
                published_at=record["published_at"],
                score=ranked_item.score,
                summary_text=summary_text,
            )
        )

    _create_or_update_digest_run(
        conn,
        digest_id=digest_id,
        user_id=user.user_id,
        context=context,
        status="completed",
        article_ids=[a.article_id for a in digest_articles],
    )

    if log_sent_event:
        store = InteractionStore(conn)
        store.log_event(
            InteractionEvent(
                user_id=user.user_id,
                event_type=EventType.DIGEST_SENT,
                digest_id=digest_id,
                metadata={
                    "window_key": context.window_key,
                    "article_count": len(digest_articles),
                },
            )
        )

    conn.commit()

    return DigestResult(
        digest_id=digest_id,
        user_id=user.user_id,
        window_key=context.window_key,
        status="completed",
        articles=digest_articles,
    )


def get_due_users(conn: sqlite3.Connection, now_utc: Optional[datetime] = None) -> List[UserProfile]:
    now = now_utc or datetime.now(timezone.utc)
    users = get_all_active_users(conn)
    due: List[UserProfile] = []

    for user in users:
        if not is_digest_due(user, now):
            continue
        context = digest_context_for_user(user, now)
        existing = _get_digest_run(conn, user.user_id, context.window_key)
        if existing and existing["status"] == "completed":
            continue
        due.append(user)

    return due


def get_latest_digest(conn: sqlite3.Connection, user_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT * FROM digest_runs
        WHERE user_id = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()
    if not row:
        return None
    return dict(row)
