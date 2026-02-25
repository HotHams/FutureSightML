#!/usr/bin/env python3
"""Train models for all formats that have sufficient data.

Usage:
    python scripts/train_all_formats.py                 # Default: 200K limit
    python scripts/train_all_formats.py --limit 500000  # Custom limit
    python scripts/train_all_formats.py --limit 0       # No limit (use all data)
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown.config import load_config
from showdown.data.database import Database
from showdown.utils.logging_config import setup_logging


async def main():
    parser = argparse.ArgumentParser(description="Train all formats")
    parser.add_argument("--limit", type=int, default=200000,
                        help="Max battles per format (0=unlimited, default=200000)")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    log = setup_logging("INFO")
    cfg = load_config(args.config)
    db_path = cfg.get("database", {}).get("path", "data/replays.db")
    db = Database(db_path)
    await db.connect()

    # All formats to train
    formats = []
    for cat, fmt_list in cfg.get("formats", {}).items():
        formats.extend(fmt_list)

    min_rating = cfg.get("training", {}).get("min_rating_filter", 1200)
    limit = args.limit if args.limit > 0 else None

    log.info("Checking replay counts for all formats...")
    log.info("Settings: min_rating=%d, limit=%s", min_rating, limit or "unlimited")

    for fmt in formats:
        count = await db.get_replay_count(fmt)
        log.info("  %s: %d replays", fmt, count)

        if count >= 500:
            log.info("  -> Training %s (both models)...", fmt)
            cmd = [
                sys.executable, "scripts/train_model.py",
                "--format", fmt, "--model", "both",
                "--min-rating", str(min_rating),
            ]
            if limit:
                cmd.extend(["--limit", str(limit)])
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parent.parent),
                capture_output=False,
            )
            if result.returncode != 0:
                log.error("  -> Training FAILED for %s (exit code %d)", fmt, result.returncode)
            else:
                log.info("  -> Training COMPLETE for %s", fmt)
        else:
            log.info("  -> Skipping %s (only %d replays, need 500+)", fmt, count)

    await db.close()
    log.info("All format training complete!")


if __name__ == "__main__":
    asyncio.run(main())
