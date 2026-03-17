# AI News Ingestion (Phase 1)

English-only RSS ingestion with full-article extraction using newspaper3k. Start with a small, reliable feed set and keep runs idempotent.

## Quick start
1) Create a virtualenv and install deps:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
2) Run the ingester (creates SQLite DB on first run):
```
python -m src.ingest
```

3) Embed articles (after ingest):
```
python -m pipeline.embed_articles
```

4) Build user interest embeddings (after interactions exist):
```
python -m pipeline.build_user_interests
```

## What it does
- Loads feed URLs from `config/feeds.json`
- Parses RSS entries, normalizes fields, and fetches full article content
- Upserts into SQLite at `data/articles.db` with a unique index on `url`
- Logs ingest counts and errors; safe to run repeatedly

## Config
- `config/feeds.json`: list of RSS URLs
- Edit the list to add/remove feeds; rerun to ingest

## Schema (SQLite)
Required fields: id, url, title, source, published_at, summary, content, ingested_at
Metadata: author, image_url, categories, source_feed_url, fetched_at, word_count, canonical_url, guid, language (fixed to "en"), topic (null), embedding (null)

## Next steps
- Add monitoring/log shipping if needed
- Switch to Postgres by replacing the SQLite connection string and DDL
- Add retry/backoff tuning per feed if you see throttling
