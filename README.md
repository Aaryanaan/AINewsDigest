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

3) End-to-end pipeline (ingest → clean → embed → interests → rank):
```
python -m pipeline.run_pipeline
```
Skip steps with flags, e.g. `python -m pipeline.run_pipeline --skip-ingest --skip-clean` when data is already present.

4) Individual steps (if running manually):
```
python -m pipeline.embed_articles
python -m pipeline.build_user_interests
python -m src.rank
```

5) Start the web digest UI (single-user dev mode):
```
uvicorn src.web_app:app --reload
```
Open `http://127.0.0.1:8000/digest`

6) Run timed digest scheduler:
```
python -m pipeline.run_digest_scheduler --once
```
Or continuous polling:
```
python -m pipeline.run_digest_scheduler --interval-seconds 60
```

## What it does
- Loads feed URLs from `config/feeds.json`
- Parses RSS entries, normalizes fields, and fetches full article content
- Upserts into SQLite at `data/articles.db` with a unique index on `url`
- Logs ingest counts and errors; safe to run repeatedly

## Config
- `config/feeds.json`: list of RSS URLs
- Edit the list to add/remove feeds; rerun to ingest
- `OPENAI_API_KEY`: required for LLM article summaries in the web digest UI

## Timed personalized summaries (MVP)
- Digest schedule uses user preferences stored in `users` (`digest_frequency`, `digest_day`, `digest_time`, `timezone`)
- Scheduler precomputes due digests into `digest_runs`
- Per-user per-window article summaries are cached in `article_summaries`
- UI loads cached summaries and can force-refresh summaries with the refresh button

## Schema (SQLite)
Required fields: id, url, title, source, published_at, summary, content, ingested_at
Metadata: author, image_url, categories, source_feed_url, fetched_at, word_count, canonical_url, guid, language (fixed to "en"), topic (null), embedding (null)

## Next steps
- Add monitoring/log shipping if needed
- Switch to Postgres by replacing the SQLite connection string and DDL
- Add retry/backoff tuning per feed if you see throttling
