"""End-to-end runner for ingest → embed → interests → rank."""

from __future__ import annotations

import argparse

from pipeline import embed_articles, build_user_interests, clean_articles
from src import interaction_ingestion
from src import ingest, rank

def run_step(enabled: bool, label: str, fn) -> None:
    if not enabled:
        print(f"[skip] {label}")
        return
    print(f"[run] {label}")
    fn()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full news pipeline")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip RSS ingest")
    parser.add_argument("--skip-clean", action="store_true", help="Skip article cleanup")
    parser.add_argument("--skip-embed", action="store_true", help="Skip article embedding")
    parser.add_argument("--skip-interests", action="store_true", help="Skip building user interest profiles")
    parser.add_argument("--skip-rank", action="store_true", help="Skip ranking")
    args = parser.parse_args()

    run_step(not args.skip_ingest, "ingest", ingest.main)
    run_step(not args.skip_clean, "clean", clean_articles.clean_articles)
    run_step(not args.skip_embed, "embed", embed_articles.main)
    run_step(True, "seed_users", interaction_ingestion.main)
    run_step(not args.skip_interests, "build_user_interests", build_user_interests.main)
    run_step(not args.skip_rank, "rank", rank.main)


if __name__ == "__main__":
    main()
