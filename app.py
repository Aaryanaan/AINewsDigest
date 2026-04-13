"""Flask UI for AINewsDigest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, redirect, url_for, abort, request
from src.db import get_connection, ensure_all_tables
from src.user_schema import get_all_active_users, get_user_by_id
from src.ranking import get_top_articles_for_user

app = Flask(__name__)


def get_db():
    conn = get_connection()
    ensure_all_tables(conn)
    return conn


def get_last_updated(conn):
    row = conn.execute(
        "SELECT MAX(published_at) AS ts FROM articles"
    ).fetchone()
    return row["ts"] if row else None


@app.context_processor
def inject_globals():
    conn = get_db()
    try:
        last_updated = get_last_updated(conn)
    finally:
        conn.close()
    return {"last_updated": last_updated}


@app.route("/")
def index():
    conn = get_db()
    users = get_all_active_users(conn)
    conn.close()
    return render_template("index.html", users=users)


@app.route("/digest/<user_id>")
def digest(user_id):
    conn = get_db()
    user = get_user_by_id(conn, user_id)
    if user is None:
        conn.close()
        abort(404)
    articles = get_top_articles_for_user(conn, user_id, top_k=user.preferences.max_articles)
    all_users = get_all_active_users(conn)
    conn.close()
    show_scores = request.args.get("debug") == "1"
    return render_template(
        "digest.html",
        user=user,
        articles=articles,
        all_users=all_users,
        show_scores=show_scores,
    )


@app.route("/articles")
def articles():
    conn = get_db()
    source_filter = request.args.get("source", "").strip()
    query = request.args.get("q", "").strip()

    sql = """
        SELECT id, title, url, source, published_at, summary, word_count
        FROM articles
    """
    params: list = []
    where: list[str] = []
    if source_filter:
        where.append("source = ?")
        params.append(source_filter)
    if query:
        where.append("(title LIKE ? OR summary LIKE ?)")
        like = f"%{query}%"
        params.extend([like, like])
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY published_at DESC LIMIT 100"

    rows = conn.execute(sql, params).fetchall()
    sources = [
        r["source"] for r in conn.execute(
            "SELECT DISTINCT source FROM articles WHERE source IS NOT NULL ORDER BY source"
        ).fetchall()
    ]
    conn.close()
    return render_template(
        "articles.html",
        articles=[dict(r) for r in rows],
        sources=sources,
        selected_source=source_filter,
        query=query,
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", code=404,
                           message="We couldn't find that page."), 404


@app.errorhandler(500)
def server_error(_e):
    return render_template("error.html", code=500,
                           message="Something went wrong on our end."), 500


if __name__ == "__main__":
    app.run(debug=True)
