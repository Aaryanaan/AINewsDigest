"""Flask UI for AINewsDigest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, redirect, url_for, abort
from src.db import get_connection, ensure_all_tables
from src.user_schema import get_all_active_users, get_user_by_id
from src.ranking import get_top_articles_for_user
from src.summarization import generate_digest
from src.llm_client import OpenAILLM

app = Flask(__name__)


def get_db():
    conn = get_connection()
    ensure_all_tables(conn)
    return conn


@app.route("/")
def index():
    conn = get_db()
    users = get_all_active_users(conn)
    conn.close()
    if len(users) == 1:
        return redirect(url_for("digest", user_id=users[0].user_id))
    return render_template("index.html", users=users)


@app.route("/digest/<user_id>")
def digest(user_id):
    conn = get_db()
    user = get_user_by_id(conn, user_id)

    if user is None:
        conn.close()
        abort(404)

    articles = get_top_articles_for_user(
        conn, user_id, top_k=user.preferences.max_articles
    )
    llm = OpenAILLM()
    digest_text = generate_digest(llm, articles)
    all_users = get_all_active_users(conn)
    conn.close()
    return render_template(
        "digest.html",
        user=user,
        articles=articles,
        digest_text=digest_text,
        all_users=all_users,
    )


@app.route("/articles")
def articles():
    conn = get_db()
    rows = conn.execute(
        """
        SELECT id, title, url, source, published_at, summary, word_count
        FROM articles
        ORDER BY published_at DESC
        LIMIT 100
        """
    ).fetchall()
    conn.close()
    return render_template("articles.html", articles=[dict(r) for r in rows])


if __name__ == "__main__":
    app.run(debug=True)
