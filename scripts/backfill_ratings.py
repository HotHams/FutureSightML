#!/usr/bin/env python3
"""Backfill rating2 (and correct rating1) for existing replays.

Re-fetches replay JSON from the Showdown API and parses both players'
pre-battle Elo ratings from the |player| lines in the battle log.

Usage:
    python scripts/backfill_ratings.py                    # Backfill all
    python scripts/backfill_ratings.py --format gen9ou    # Backfill one format
    python scripts/backfill_ratings.py --batch-size 500   # Custom batch size
    python scripts/backfill_ratings.py --dry-run           # Preview without updating
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown.config import load_config
from showdown.data.database import Database
from showdown.scraper.replay_parser import ReplayParser

log = logging.getLogger("backfill_ratings")


async def fetch_replay(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                       base_url: str, replay_id: str, delay: float) -> dict | None:
    url = f"{base_url}/{replay_id}.json"
    try:
        async with semaphore:
            await asyncio.sleep(delay)
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
    except Exception as e:
        log.debug("Failed to fetch %s: %s", replay_id, e)
        return None


async def main():
    parser_arg = argparse.ArgumentParser(description="Backfill ratings from replay logs")
    parser_arg.add_argument("--format", type=str, default=None, help="Only backfill this format")
    parser_arg.add_argument("--batch-size", type=int, default=200, help="Replays per batch")
    parser_arg.add_argument("--max-concurrent", type=int, default=15, help="Max concurrent requests")
    parser_arg.add_argument("--delay-ms", type=int, default=50, help="Delay between requests (ms)")
    parser_arg.add_argument("--dry-run", action="store_true", help="Preview without updating DB")
    parser_arg.add_argument("--config", type=str, default=None)
    args = parser_arg.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    db_path = cfg.get("database", {}).get("path", "data/replays.db")
    base_url = cfg.get("scraper", {}).get("replay_base_url", "https://replay.pokemonshowdown.com")

    db = Database(db_path)
    await db.connect()
    replay_parser = ReplayParser()

    # Get replay IDs that need backfill
    query = "SELECT id FROM replays WHERE rating2 IS NULL"
    params = []
    if args.format:
        query += " AND format = ?"
        params.append(args.format)
    query += " ORDER BY upload_time DESC"

    cursor = await db.conn.execute(query, params)
    rows = await cursor.fetchall()
    replay_ids = [r[0] for r in rows]
    log.info("Found %d replays needing rating backfill", len(replay_ids))

    if not replay_ids:
        log.info("Nothing to backfill!")
        await db.close()
        return

    semaphore = asyncio.Semaphore(args.max_concurrent)
    delay = args.delay_ms / 1000.0
    timeout = aiohttp.ClientTimeout(total=30)

    stats = {"updated": 0, "failed": 0, "skipped": 0, "both_found": 0}
    start_time = time.time()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for batch_start in range(0, len(replay_ids), args.batch_size):
            batch = replay_ids[batch_start:batch_start + args.batch_size]

            # Fetch all replays in batch concurrently
            tasks = [
                fetch_replay(session, semaphore, base_url, rid, delay)
                for rid in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Parse and update
            for rid, result in zip(batch, results):
                if isinstance(result, Exception) or result is None:
                    stats["failed"] += 1
                    continue

                log_text = result.get("log", "")
                if not log_text:
                    stats["failed"] += 1
                    continue

                battle = replay_parser.parse(log_text, "")
                r1 = battle.rating1
                r2 = battle.rating2

                if r1 is None and r2 is None:
                    stats["skipped"] += 1
                    continue

                if r1 is not None and r2 is not None:
                    stats["both_found"] += 1

                if not args.dry_run:
                    await db.conn.execute(
                        "UPDATE replays SET rating1 = COALESCE(?, rating1), rating2 = ? WHERE id = ?",
                        (r1, r2, rid),
                    )

                stats["updated"] += 1

            if not args.dry_run:
                await db.conn.commit()

            elapsed = time.time() - start_time
            processed = batch_start + len(batch)
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (len(replay_ids) - processed) / rate if rate > 0 else 0
            log.info(
                "Progress: %d/%d (%.1f%%) | Updated: %d | Both: %d | Failed: %d | %.0f/s | ETA: %.0fs",
                processed, len(replay_ids), processed / len(replay_ids) * 100,
                stats["updated"], stats["both_found"], stats["failed"],
                rate, remaining,
            )

    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info("BACKFILL COMPLETE in %.0f seconds", elapsed)
    log.info("Updated: %d | Both ratings found: %d | Failed: %d | Skipped: %d",
             stats["updated"], stats["both_found"], stats["failed"], stats["skipped"])
    log.info("=" * 60)

    # Verify
    if not args.dry_run:
        cur = await db.conn.execute("SELECT COUNT(*) FROM replays WHERE rating2 IS NOT NULL")
        filled = (await cur.fetchone())[0]
        cur = await db.conn.execute("SELECT COUNT(*) FROM replays WHERE rating2 IS NULL")
        still_null = (await cur.fetchone())[0]
        log.info("Post-backfill: %d with rating2, %d still NULL", filled, still_null)

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
