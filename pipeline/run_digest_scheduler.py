"""Scheduler loop for timed digest precompute and caching."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from src.db import ensure_all_tables, get_connection
from src.digest_service import build_digest_for_user, get_due_users


def run_once() -> int:
    conn = get_connection()
    ensure_all_tables(conn)

    due_users = get_due_users(conn, now_utc=datetime.now(timezone.utc))
    if not due_users:
        print("[digest-scheduler] no due users")
        conn.close()
        return 0

    for user in due_users:
        print(f"[digest-scheduler] generating digest for {user.user_id} ({user.email})")
        result = build_digest_for_user(conn, user=user, force_refresh=False, log_sent_event=True)
        print(
            f"[digest-scheduler] completed digest={result.digest_id} "
            f"window={result.window_key} articles={len(result.articles)}"
        )

    conn.close()
    return len(due_users)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run timed digest scheduler")
    parser.add_argument("--once", action="store_true", help="Run one scheduler cycle and exit")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=60,
        help="Polling interval between scheduler cycles",
    )
    args = parser.parse_args()

    if args.once:
        run_once()
        return

    interval = max(15, args.interval_seconds)
    print(f"[digest-scheduler] running loop every {interval}s")

    while True:
        try:
            run_once()
        except Exception as exc:
            print(f"[digest-scheduler] error: {exc}")
        time.sleep(interval)


if __name__ == "__main__":
    main()
