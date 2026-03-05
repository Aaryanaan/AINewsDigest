import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "data/articles.db"
model = SentenceTransformer("all-MiniLM-L6-v2")
conn = sqlite3.connect(DB_PATH)
rows = conn.execute(
    "SELECT id, content FROM articles WHERE embedding IS NULL"
).fetchall()
for article_id, content in rows:
    if not content:
        continue
    embedding = model.encode(content)
    conn.execute(
        "UPDATE articles SET embedding=? WHERE id=?",
        (embedding.tobytes(), article_id),
    )
conn.commit()