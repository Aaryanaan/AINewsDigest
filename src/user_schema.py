"""
user_schema.py — User profile schema + SQLite persistence
==========================================================
Pydantic models define the application-layer representation.
Helper functions handle reads/writes against the shared SQLite DB
(see db.py for table creation).

Works with Kevin's article ingestion pipeline — both share the same
database via `db.get_connection()`.
"""

from __future__ import annotations

import json
import uuid
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, EmailStr, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DigestFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"


class DayOfWeek(str, Enum):
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class UserPreferences(BaseModel):
    """Digest delivery & content preferences."""
    digest_frequency: DigestFrequency = DigestFrequency.WEEKLY
    digest_day: DayOfWeek = DayOfWeek.MONDAY
    digest_time: str = Field(default="08:00")
    timezone: str = Field(default="America/New_York")
    max_articles: int = Field(default=10, ge=1, le=50)
    preferred_sources: list[str] = Field(default_factory=list)
    blocked_sources: list[str] = Field(default_factory=list)
    language: str = Field(default="en")
    reading_level: Optional[str] = None


class UserInterestProfile(BaseModel):
    """Current interest representation used by the ranking module."""
    interest_embedding: Optional[list[float]] = None
    interest_topics: list[str] = Field(default_factory=list)
    last_profile_update_at: Optional[datetime] = None


class UserProfile(BaseModel):
    """
    Top-level user schema.

    Downstream components (ranking, digest generation, scheduling,
    evaluation) import and rely on this model.
    """
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    display_name: Optional[str] = None
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    interests: UserInterestProfile = Field(default_factory=UserInterestProfile)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("email")
    @classmethod
    def lowercase_email(cls, v: str) -> str:
        return v.lower()


# ---------------------------------------------------------------------------
# SQLite CRUD helpers (raw sqlite3 to match Kevin's approach)
# ---------------------------------------------------------------------------

def upsert_user(conn: sqlite3.Connection, user: UserProfile) -> None:
    """Insert or update a user row in the shared DB."""
    conn.execute(
        """
        INSERT INTO users (
            user_id, email, display_name,
            digest_frequency, digest_day, digest_time, timezone,
            max_articles, preferred_sources, blocked_sources,
            language, reading_level,
            interest_embedding, interest_topics, last_profile_update_at,
            is_active, created_at, updated_at
        ) VALUES (
            :user_id, :email, :display_name,
            :digest_frequency, :digest_day, :digest_time, :timezone,
            :max_articles, :preferred_sources, :blocked_sources,
            :language, :reading_level,
            :interest_embedding, :interest_topics, :last_profile_update_at,
            :is_active, :created_at, :updated_at
        )
        ON CONFLICT(email) DO UPDATE SET
            display_name=excluded.display_name,
            digest_frequency=excluded.digest_frequency,
            digest_day=excluded.digest_day,
            digest_time=excluded.digest_time,
            timezone=excluded.timezone,
            max_articles=excluded.max_articles,
            preferred_sources=excluded.preferred_sources,
            blocked_sources=excluded.blocked_sources,
            language=excluded.language,
            reading_level=excluded.reading_level,
            interest_embedding=excluded.interest_embedding,
            interest_topics=excluded.interest_topics,
            last_profile_update_at=excluded.last_profile_update_at,
            is_active=excluded.is_active,
            updated_at=excluded.updated_at;
        """,
        _user_to_row(user),
    )
    conn.commit()


def get_user_by_email(conn: sqlite3.Connection, email: str) -> Optional[UserProfile]:
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email.lower(),)
    ).fetchone()
    return _row_to_user(row) if row else None


def get_user_by_id(conn: sqlite3.Connection, user_id: str) -> Optional[UserProfile]:
    row = conn.execute(
        "SELECT * FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()
    return _row_to_user(row) if row else None


def get_all_active_users(conn: sqlite3.Connection) -> List[UserProfile]:
    rows = conn.execute(
        "SELECT * FROM users WHERE is_active = 1"
    ).fetchall()
    return [_row_to_user(r) for r in rows]


# ---------------------------------------------------------------------------
# Row ↔ Pydantic converters
# ---------------------------------------------------------------------------

def _user_to_row(u: UserProfile) -> dict:
    return {
        "user_id": u.user_id,
        "email": u.email,
        "display_name": u.display_name,
        "digest_frequency": u.preferences.digest_frequency.value,
        "digest_day": u.preferences.digest_day.value,
        "digest_time": u.preferences.digest_time,
        "timezone": u.preferences.timezone,
        "max_articles": u.preferences.max_articles,
        "preferred_sources": json.dumps(u.preferences.preferred_sources),
        "blocked_sources": json.dumps(u.preferences.blocked_sources),
        "language": u.preferences.language,
        "reading_level": u.preferences.reading_level,
        "interest_embedding": json.dumps(u.interests.interest_embedding)
            if u.interests.interest_embedding else None,
        "interest_topics": json.dumps(u.interests.interest_topics),
        "last_profile_update_at": u.interests.last_profile_update_at.isoformat()
            if u.interests.last_profile_update_at else None,
        "is_active": int(u.is_active),
        "created_at": u.created_at.isoformat(),
        "updated_at": u.updated_at.isoformat(),
    }


def _row_to_user(row: sqlite3.Row) -> UserProfile:
    return UserProfile(
        user_id=row["user_id"],
        email=row["email"],
        display_name=row["display_name"],
        preferences=UserPreferences(
            digest_frequency=row["digest_frequency"],
            digest_day=row["digest_day"],
            digest_time=row["digest_time"],
            timezone=row["timezone"],
            max_articles=row["max_articles"],
            preferred_sources=json.loads(row["preferred_sources"] or "[]"),
            blocked_sources=json.loads(row["blocked_sources"] or "[]"),
            language=row["language"],
            reading_level=row["reading_level"],
        ),
        interests=UserInterestProfile(
            interest_embedding=json.loads(row["interest_embedding"])
                if row["interest_embedding"] else None,
            interest_topics=json.loads(row["interest_topics"] or "[]"),
            last_profile_update_at=row["last_profile_update_at"],
        ),
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from db import get_connection, ensure_all_tables

    conn = get_connection()
    ensure_all_tables(conn)

    user = UserProfile(
        email="aaryan.nanekar@gmail.com",
        display_name="Aaryan",
        preferences=UserPreferences(
            preferred_sources=["techcrunch", "arxiv"],
            max_articles=15,
        ),
        interests=UserInterestProfile(
            interest_topics=["llm", "robotics", "computer-vision"],
        ),
    )

    upsert_user(conn, user)
    fetched = get_user_by_email(conn, "aaryan.nanekar@gmail.com")
    print("Round-tripped user from SQLite:")
    print(fetched.model_dump_json(indent=2))
    conn.close()
