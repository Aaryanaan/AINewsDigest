import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import feedparser
from dateutil import parser as dateparser
import requests
import httpx
from newspaper import Article
from tenacity import retry, stop_after_attempt, wait_exponential
import tldextract

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "articles.db"
FEEDS_PATH = ROOT / "config" / "feeds.json"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def load_feeds(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [url.strip() for url in json.load(f) if url.strip()]


def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            title TEXT,
            url TEXT NOT NULL,
            source TEXT,
            published_at TEXT,
            summary TEXT,
            content TEXT,
            topic TEXT,
            embedding BLOB,
            ingested_at TEXT NOT NULL,
            author TEXT,
            image_url TEXT,
            categories TEXT,
            source_feed_url TEXT,
            fetched_at TEXT,
            word_count INTEGER,
            canonical_url TEXT,
            guid TEXT,
            language TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
        """
    )
    conn.commit()


def parse_datetime(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return dateparser.parse(value).isoformat()
    except Exception:
        return None


def normalize_source(link: str, feed_title: Optional[str]) -> str:
    if feed_title:
        return feed_title
    parts = tldextract.extract(link)
    domain = ".".join(part for part in [parts.domain, parts.suffix] if part)
    return domain or "unknown"


def build_id(url: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))


def collect_categories(entry) -> Optional[str]:
    tags = entry.get("tags") or []
    terms = []
    for tag in tags:
        term = tag.get("term") if isinstance(tag, dict) else getattr(tag, "term", None)
        if term:
            terms.append(term)
    if not terms:
        return None
    return ",".join(terms)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def fetch_full_content(url: str) -> Dict[str, Optional[str]]:
    html, final_url = fetch_html_with_fallback(url)

    article = Article(final_url, language="en")
    article.set_html(html)
    article.parse()
    text = article.text or None
    author = None
    if article.authors:
        author = ",".join(article.authors)
    word_count = len(text.split()) if text else None
    return {
        "title": article.title or None,
        "content": text,
        "author": author,
        "image_url": article.top_image or None,
        "canonical_url": article.canonical_link or None,
        "word_count": word_count,
    }


def fetch_html_with_fallback(url: str) -> Tuple[str, str]:
    # Try requests first
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15, allow_redirects=True)
    if resp.status_code == 403:
        # Try httpx with a different TLS stack; some sites are UA/JA3 picky
        with httpx.Client(follow_redirects=True, headers={"User-Agent": USER_AGENT}, timeout=15) as client:
            resp = client.get(url)
    resp.raise_for_status()
    return resp.text, str(resp.url)


def upsert_article(conn: sqlite3.Connection, record: Dict[str, Optional[str]]) -> None:
    conn.execute(
        """
        INSERT INTO articles (
            id, url, title, source, published_at, summary, content, topic, embedding,
            ingested_at, author, image_url, categories, source_feed_url, fetched_at,
            word_count, canonical_url, guid, language
        ) VALUES (
            :id, :url, :title, :source, :published_at, :summary, :content, NULL, NULL,
            :ingested_at, :author, :image_url, :categories, :source_feed_url, :fetched_at,
            :word_count, :canonical_url, :guid, :language
        )
        ON CONFLICT(url) DO UPDATE SET
            title=excluded.title,
            source=excluded.source,
            published_at=excluded.published_at,
            summary=excluded.summary,
            content=excluded.content,
            ingested_at=excluded.ingested_at,
            author=excluded.author,
            image_url=excluded.image_url,
            categories=excluded.categories,
            source_feed_url=excluded.source_feed_url,
            fetched_at=excluded.fetched_at,
            word_count=excluded.word_count,
            canonical_url=excluded.canonical_url,
            guid=excluded.guid,
            language=excluded.language;
        """,
        record,
    )


def ingest_feed(feed_url: str, conn: sqlite3.Connection) -> int:
    feed = feedparser.parse(feed_url)
    source = normalize_source(feed_url, getattr(feed, "feed", {}).get("title"))
    ingested = 0
    for entry in feed.entries:
        url = entry.get("link")
        if not url:
            continue
        guid = entry.get("id") or entry.get("guid") or None
        published_at = parse_datetime(entry.get("published") or entry.get("updated"))
        summary = entry.get("summary") or None
        categories = collect_categories(entry)
        fetched_at = datetime.utcnow().isoformat()

        full = {}
        try:
            full = fetch_full_content(url)
        except Exception as exc:  # keep going even if fetch fails
            print(f"[warn] full-content fetch failed for {url}: {exc}")

        title = entry.get("title") or full.get("title") or None
        content = full.get("content")
        if not content:
            # fall back to summary if no content was extracted
            content = summary

        record = {
            "id": build_id(url),
            "url": url,
            "title": title,
            "source": source,
            "published_at": published_at,
            "summary": summary,
            "content": content,
            "ingested_at": fetched_at,
            "author": full.get("author"),
            "image_url": full.get("image_url"),
            "categories": categories,
            "source_feed_url": feed_url,
            "fetched_at": fetched_at,
            "word_count": full.get("word_count"),
            "canonical_url": full.get("canonical_url"),
            "guid": guid,
            "language": "en",
        }

        upsert_article(conn, record)
        ingested += 1

    conn.commit()
    return ingested


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    feeds = load_feeds(FEEDS_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        ensure_db(conn)
        total = 0
        for feed_url in feeds:
            try:
                count = ingest_feed(feed_url, conn)
                total += count
                print(f"[ok] Ingested {count} articles from {feed_url}")
            except Exception as exc:
                print(f"[error] Failed ingest for {feed_url}: {exc}")
        print(f"[done] Total ingested (including updates): {total}")


if __name__ == "__main__":
    main()
