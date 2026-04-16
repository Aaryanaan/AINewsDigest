"""FastAPI app for viewing timed personalized news digests."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from src.db import ensure_all_tables, get_connection
from src.digest_service import build_digest_for_user, get_latest_digest
from src.interaction_ingestion import EventType, InteractionEvent, InteractionStore
from src.user_schema import get_all_active_users

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Jinja2Templates(directory=str(ROOT / "templates"))

app = FastAPI(title="AI News Digest")


def _get_dev_user(conn):
    users = get_all_active_users(conn)
    if not users:
        return None
    return users[0]


@app.get("/")
def home() -> RedirectResponse:
    return RedirectResponse(url="/digest", status_code=302)


@app.get("/digest")
def digest_page(request: Request):
    conn = get_connection()
    ensure_all_tables(conn)
    user = _get_dev_user(conn)
    if user is None:
        conn.close()
        return TEMPLATES.TemplateResponse(
            request,
            "digest.html",
            {
                "has_user": False,
                "message": "No active user found. Seed/create a user profile first.",
                "articles": [],
            },
        )

    digest = build_digest_for_user(conn, user=user, force_refresh=False)

    store = InteractionStore(conn)
    store.log_event(
        InteractionEvent(
            user_id=user.user_id,
            event_type=EventType.DIGEST_OPENED,
            digest_id=digest.digest_id,
            metadata={"window_key": digest.window_key},
        )
    )

    latest = get_latest_digest(conn, user.user_id)
    conn.close()

    return TEMPLATES.TemplateResponse(
        request,
        "digest.html",
        {
            "has_user": True,
            "user": user,
            "digest": digest,
            "latest": latest,
            "articles": digest.articles,
            "message": None,
        },
    )


@app.post("/digest/refresh")
def refresh_digest(force_refresh: str = Form(default="1")) -> RedirectResponse:
    conn = get_connection()
    ensure_all_tables(conn)
    user = _get_dev_user(conn)
    if user is not None:
        should_force = force_refresh == "1"
        build_digest_for_user(conn, user=user, force_refresh=should_force)
    conn.close()
    return RedirectResponse(url="/digest", status_code=303)
