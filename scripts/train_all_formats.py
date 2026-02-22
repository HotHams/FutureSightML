#!/usr/bin/env python3
"""Train models for all formats that have sufficient data."""

import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown.config import load_config
from showdown.data.database import Database
from showdown.utils.logging_config import setup_logging


async def main():
    log = setup_logging("INFO")
    cfg = load_config()
    db_path = cfg.get("database", {}).get("path", "data/replays.db")
    db = Database(db_path)
    await db.connect()

    # All formats to train
    formats = []
    for cat, fmt_list in cfg.get("formats", {}).items():
        formats.extend(fmt_list)

    log.info("Checking replay counts for all formats...")
    for fmt in formats:
        count = await db.get_replay_count(fmt)
        log.info("  %s: %d replays", fmt, count)

        if count >= 500:
            log.info("  -> Training %s (both models)...", fmt)
            result = subprocess.run(
                [sys.executable, "scripts/train_model.py",
                 "--format", fmt, "--model", "both", "--min-rating", "1000"],
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
