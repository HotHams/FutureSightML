#!/usr/bin/env python3
"""Scrape Smogon monthly usage statistics.

Usage:
    python scripts/scrape_stats.py --format gen9ou
    python scripts/scrape_stats.py --all --months 6
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown.config import load_config
from showdown.data.database import Database
from showdown.scraper.stats_scraper import StatsScraper
from showdown.utils.logging_config import setup_logging


async def main():
    parser = argparse.ArgumentParser(description="Scrape Smogon usage statistics")
    parser.add_argument("--format", type=str, help="Format to scrape (e.g. gen9ou)")
    parser.add_argument("--all", action="store_true", help="Scrape all configured formats")
    parser.add_argument("--months", type=int, default=6, help="Number of recent months to scrape")
    parser.add_argument("--ratings", type=str, default="1825",
                        help="Comma-separated rating thresholds (e.g. 1500,1695,1825)")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    log = setup_logging("INFO")
    cfg = load_config(args.config)

    db_path = cfg.get("database", {}).get("path", "data/showdown.db")
    db = Database(db_path)
    await db.connect()

    scraper = StatsScraper(db=db)

    formats = []
    if args.all:
        for fmt_list in cfg.get("formats", {}).values():
            formats.extend(fmt_list)
    elif args.format:
        formats = [args.format]
    else:
        parser.error("Specify --format or --all")

    ratings = [int(r.strip()) for r in args.ratings.split(",")]
    months = StatsScraper._recent_months(args.months)

    total = 0
    try:
        for fmt in formats:
            log.info("Scraping stats for %s (%d months, ratings=%s)", fmt, len(months), ratings)
            count = await scraper.scrape_format(fmt, months=months, rating_thresholds=ratings)
            total += count
            log.info("Scraped %d stat files for %s", count, fmt)
    finally:
        await scraper.close()
        await db.close()

    log.info("Done. Total stat files scraped: %d", total)


if __name__ == "__main__":
    asyncio.run(main())
