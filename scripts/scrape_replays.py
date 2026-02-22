#!/usr/bin/env python3
"""Scrape Pokemon Showdown replays for training data.

Usage:
    python scripts/scrape_replays.py --format gen9ou --pages 100
    python scripts/scrape_replays.py --all --pages 50
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown.config import load_config
from showdown.data.database import Database
from showdown.scraper.replay_scraper import ReplayScraper
from showdown.utils.logging_config import setup_logging


async def main():
    parser = argparse.ArgumentParser(description="Scrape Pokemon Showdown replays")
    parser.add_argument("--format", type=str, help="Format to scrape (e.g. gen9ou)")
    parser.add_argument("--all", action="store_true", help="Scrape all configured formats")
    parser.add_argument("--pages", type=int, default=None, help="Max pages per format (default: from config)")
    parser.add_argument("--min-rating", type=int, default=None, help="Minimum player rating")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    log = setup_logging("INFO")
    cfg = load_config(args.config)

    scraper_cfg = cfg.get("scraper", {})
    db_path = cfg.get("database", {}).get("path", "data/showdown.db")
    min_rating = args.min_rating or scraper_cfg.get("min_rating", 1500)
    max_pages = args.pages or scraper_cfg.get("max_pages_per_format", 100)

    db = Database(db_path)
    await db.connect()

    scraper = ReplayScraper(
        db=db,
        base_url=scraper_cfg.get("replay_base_url", "https://replay.pokemonshowdown.com"),
        max_concurrent=scraper_cfg.get("max_concurrent_requests", 10),
        request_delay_ms=scraper_cfg.get("request_delay_ms", 100),
        min_rating=min_rating,
    )

    formats = []
    if args.all:
        for fmt_list in cfg.get("formats", {}).values():
            formats.extend(fmt_list)
    elif args.format:
        formats = [args.format]
    else:
        parser.error("Specify --format or --all")

    total_stats = {"fetched": 0, "parsed": 0, "skipped": 0, "errors": 0}

    try:
        for fmt in formats:
            log.info("=" * 60)
            log.info("Scraping format: %s", fmt)
            log.info("=" * 60)

            existing = await db.get_replay_count(fmt)
            log.info("Existing replays for %s: %d", fmt, existing)

            stats = await scraper.scrape_format(fmt, max_pages=max_pages)
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

            log.info("Format %s done: %s", fmt, stats)
    finally:
        await scraper.close()
        await db.close()

    log.info("=" * 60)
    log.info("SCRAPING COMPLETE")
    log.info("Total: %s", total_stats)
    log.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
