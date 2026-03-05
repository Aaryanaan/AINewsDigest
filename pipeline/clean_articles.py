import re
import sqlite3
from pathlib import Path

DB_PATH = Path("data/articles.db")

def clean_text(text):
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xa0", " ")
    text = text.strip()
    return text

def clean_articles():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, content FROM articles"
    ).fetchall()
    for article_id, content in rows:
        cleaned = clean_text(content)
        conn.execute(
            "UPDATE articles SET content=? WHERE id=?",
            (cleaned, article_id),
        )
    conn.commit()