#!/usr/bin/env python3
"""Import Pokemon Showdown replays from the HolidayOugi HuggingFace dataset.

Downloads Parquet files for specified formats (Gen 1-9), parses the battle logs,
and inserts into our SQLite database.

Dataset: https://huggingface.co/datasets/HolidayOugi/pokemon-showdown-replays

Usage:
    python scripts/import_huggingface.py                    # Import all configured formats
    python scripts/import_huggingface.py --format gen9ou    # Import one format
    python scripts/import_huggingface.py --gen 4            # Import all Gen 4 formats
    python scripts/import_huggingface.py --min-rating 1200  # Only rated games >= 1200
    python scripts/import_huggingface.py --cleanup          # Delete parquet files after import
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown.config import load_config
from showdown.data.database import Database
from showdown.scraper.replay_parser import ReplayParser

log = logging.getLogger("import_huggingface")

# Mapping from our format IDs to HuggingFace file base names.
# Files are named like "[Gen 9] OU.parquet" or "[Gen 9] OU_part1.parquet"
# Only include formats that have matching parquet files in the HF dataset.
FORMAT_TO_HF_FILES = {
    # ── Gen 1 ──
    "gen1ou": ["[Gen 1] OU"],
    "gen1uu": ["[Gen 1] UU"],
    "gen1ubers": ["[Gen 1] UBERS"],
    "gen1nu": ["[Gen 1] NU"],
    "gen1pu": ["[Gen 1] PU"],
    "gen1lc": ["[Gen 1] LC"],
    # ── Gen 2 ──
    "gen2ou": ["[Gen 2] OU"],
    "gen2uu": ["[Gen 2] UU"],
    "gen2ubers": ["[Gen 2] UBERS"],
    "gen2nu": ["[Gen 2] NU"],
    "gen2lc": ["[Gen 2] LC"],
    # ── Gen 3 ──
    "gen3ou": ["[Gen 3] OU"],
    "gen3uu": ["[Gen 3] UU"],
    "gen3ubers": ["[Gen 3] UBERS"],
    "gen3nu": ["[Gen 3] NU"],
    "gen3ru": ["[Gen 3] RU"],
    "gen3lc": ["[Gen 3] LC"],
    "gen3monotype": ["[Gen 3] MONOTYPE"],
    "gen31v1": ["[Gen 3] 1V1"],
    "gen3doublesou": ["[Gen 3] DOUBLES OU"],
    # ── Gen 4 ──
    "gen4ou": ["[Gen 4] OU"],
    "gen4uu": ["[Gen 4] UU"],
    "gen4ubers": ["[Gen 4] UBERS"],
    "gen4nu": ["[Gen 4] NU"],
    "gen4ru": ["[Gen 4] RU"],
    "gen4lc": ["[Gen 4] LC"],
    "gen4monotype": ["[Gen 4] MONOTYPE"],
    "gen41v1": ["[Gen 4] 1V1"],
    "gen4ag": ["[Gen 4] ANYTHINGGOES"],
    "gen4doublesou": ["[Gen 4] DOUBLES OU"],
    "gen4vgc2010": ["[Gen 4] VGC 2010"],
    # ── Gen 5 ──
    "gen5ou": ["[Gen 5] OU"],
    "gen5uu": ["[Gen 5] UU"],
    "gen5ubers": ["[Gen 5] UBERS"],
    "gen5nu": ["[Gen 5] NU"],
    "gen5ru": ["[Gen 5] RU"],
    "gen5lc": ["[Gen 5] LC"],
    "gen5monotype": ["[Gen 5] MONOTYPE"],
    "gen51v1": ["[Gen 5] 1V1"],
    "gen5ag": ["[Gen 5] ANYTHINGGOES"],
    "gen5doublesou": ["[Gen 5] DOUBLES OU"],
    "gen5vgc2012": ["[Gen 5] VGC 2012"],
    "gen5vgc2011": ["[Gen 5] VGC 2011"],
    # ── Gen 6 ──
    "gen6ou": ["[Gen 6] OU"],
    "gen6uu": ["[Gen 6] UU"],
    "gen6ubers": ["[Gen 6] UBERS"],
    "gen6nu": ["[Gen 6] NU"],
    "gen6ru": ["[Gen 6] RU"],
    "gen6pu": ["[Gen 6] PU"],
    "gen6lc": ["[Gen 6] LC"],
    "gen6monotype": ["[Gen 6] MONOTYPE"],
    "gen61v1": ["[Gen 6] 1V1"],
    "gen6ag": ["[Gen 6] ANYTHINGGOES"],
    "gen6battlespotsingles": ["[Gen 6] BATTLE SPOT SINGLES"],
    "gen6doublesou": ["[Gen 6] DOUBLES OU"],
    "gen6vgc2016": ["[Gen 6] VGC 2016"],
    "gen6vgc2015": ["[Gen 6] VGC 2015"],
    "gen6vgc2014": ["[Gen 6] VGC 2014"],
    "gen6battlespotdoubles": ["[Gen 6] BATTLE SPOT DOUBLES"],
    # ── Gen 7 ──
    "gen7ou": ["[Gen 7] OU"],
    "gen7uu": ["[Gen 7] UU"],
    "gen7ubers": ["[Gen 7] UBERS"],
    "gen7nu": ["[Gen 7] NU"],
    "gen7ru": ["[Gen 7] RU"],
    "gen7pu": ["[Gen 7] PU"],
    "gen7lc": ["[Gen 7] LC"],
    "gen7monotype": ["[Gen 7] MONOTYPE"],
    "gen71v1": ["[Gen 7] 1V1"],
    "gen7ag": ["[Gen 7] ANYTHINGGOES"],
    "gen7zu": ["[Gen 7] ZU"],
    "gen7battlespotsingles": ["[Gen 7] BATTLE SPOT SINGLES"],
    "gen7doublesou": ["[Gen 7] DOUBLES OU"],
    "gen7doublesuu": ["[Gen 7] DOUBLES UU"],
    "gen7vgc2019": ["[Gen 7] VGC 2019"],
    "gen7vgc2018": ["[Gen 7] VGC 2018"],
    "gen7vgc2017": ["[Gen 7] VGC 2017"],
    "gen7battlespotdoubles": ["[Gen 7] BATTLE SPOT DOUBLES"],
    # ── Gen 8 ──
    "gen8ou": ["[Gen 8] OU"],
    "gen8uu": ["[Gen 8] UU"],
    "gen8ubers": ["[Gen 8] UBERS"],
    "gen8nu": ["[Gen 8] NU"],
    "gen8ru": ["[Gen 8] RU"],
    "gen8pu": ["[Gen 8] PU"],
    "gen8lc": ["[Gen 8] LC"],
    "gen8monotype": ["[Gen 8] MONOTYPE"],
    "gen81v1": ["[Gen 8] 1V1"],
    "gen8ag": ["[Gen 8] ANYTHINGGOES"],
    "gen8zu": ["[Gen 8] ZU"],
    "gen8nationaldex": ["[Gen 8] NATIONALDEX OU"],
    "gen8nationaldexuu": ["[Gen 8] NATIONALDEX UU"],
    "gen8nationaldexubers": ["[Gen 8] NATIONALDEX UBERS"],
    "gen8nationaldexmonotype": ["[Gen 8] NATIONALDEX MONOTYPE"],
    "gen8nationaldexag": ["[Gen 8] NATIONALDEX AG"],
    "gen8battlestadiumsingles": ["[Gen 8] BATTLE STADIUM SINGLES"],
    "gen8ubersuu": ["[Gen 8] UBERS UU"],
    "gen8doublesou": ["[Gen 8] DOUBLES OU"],
    "gen8doublesuu": ["[Gen 8] DOUBLES UU"],
    "gen8vgc2022": ["[Gen 8] VGC 2022"],
    "gen8vgc2021": ["[Gen 8] VGC 2021"],
    "gen8vgc2020": ["[Gen 8] VGC 2020"],
    "gen8battlestadiumdoubles": ["[Gen 8] BATTLE STADIUM DOUBLES"],
    "gen8nationaldexdoubles": ["[Gen 8] NATIONALDEX DOUBLES"],
    "gen82v2doubles": ["[Gen 8] 2V2 DOUBLES"],
    # ── Gen 9 ──
    "gen9ou": ["[Gen 9] OU"],
    "gen9uu": ["[Gen 9] UU"],
    "gen9ubers": ["[Gen 9] UBERS"],
    "gen9nu": ["[Gen 9] NU"],
    "gen9ru": ["[Gen 9] RU"],
    "gen9pu": ["[Gen 9] PU"],
    "gen9lc": ["[Gen 9] LC"],
    "gen9monotype": ["[Gen 9] MONOTYPE"],
    "gen91v1": ["[Gen 9] 1V1"],
    "gen9ag": ["[Gen 9] ANYTHINGGOES"],
    "gen9zu": ["[Gen 9] ZU"],
    "gen9nationaldex": ["[Gen 9] NATIONALDEX OU"],
    "gen9nationaldexuu": ["[Gen 9] NATIONALDEX UU"],
    "gen9nationaldexubers": ["[Gen 9] NATIONALDEX UBERS"],
    "gen9nationaldexmonotype": ["[Gen 9] NATIONALDEX MONOTYPE"],
    "gen9nationaldexag": ["[Gen 9] NATIONALDEX AG"],
    "gen9battlestadiumsingles": ["[Gen 9] BATTLE STADIUM SINGLES"],
    "gen9ubersuu": ["[Gen 9] UBERS UU"],
    "gen9cap": ["[Gen 9] CAP"],
    "gen9vgc2026regf": ["[Gen 9] VGC 2026"],
    "gen9doublesou": ["[Gen 9] DOUBLES OU"],
    "gen9doublesuu": ["[Gen 9] DOUBLES UU"],
    "gen9vgc2025regg": ["[Gen 9] VGC 2025"],
    "gen9battlestadiumdoubles": ["[Gen 9] BATTLE STADIUM DOUBLES"],
    "gen9nationaldexdoubles": ["[Gen 9] NATIONALDEX DOUBLES"],
    "gen92v2doubles": ["[Gen 9] 2V2 DOUBLES"],
}

HF_BASE_URL = "https://huggingface.co/datasets/HolidayOugi/pokemon-showdown-replays/resolve/main"
DOWNLOAD_DIR = Path("data/raw/huggingface")


def discover_parquet_urls(base_name: str, max_parts: int = 30) -> list[tuple[str, str]]:
    """Build list of (url, local_filename) tuples for a format's Parquet files."""
    from urllib.request import urlopen, Request
    from urllib.parse import quote
    from urllib.error import HTTPError

    results = []

    # Try single file first
    filename = f"{base_name}.parquet"
    url = f"{HF_BASE_URL}/{quote(filename)}"
    try:
        req = Request(url, method="HEAD")
        resp = urlopen(req, timeout=10)
        if resp.status == 200:
            results.append((url, filename))
            return results
    except (HTTPError, Exception):
        pass

    # Try multipart files
    for i in range(1, max_parts + 1):
        filename = f"{base_name}_part{i}.parquet"
        url = f"{HF_BASE_URL}/{quote(filename)}"
        try:
            req = Request(url, method="HEAD")
            resp = urlopen(req, timeout=10)
            if resp.status == 200:
                results.append((url, filename))
        except (HTTPError, Exception):
            break

    return results


def download_file(url: str, dest: Path) -> Path:
    """Download a file with progress reporting."""
    if dest.exists():
        log.info("Already downloaded: %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s ...", url.split("/")[-1])

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded / total_size * 100)
            if block_num % 100 == 0:
                log.info("  %.1f%% (%d MB / %d MB)", pct, downloaded // 1e6, total_size // 1e6)

    urlretrieve(url, str(dest), reporthook=progress)
    log.info("Downloaded: %s", dest.name)
    return dest


async def import_parquet(
    db: Database,
    parquet_path: Path,
    format_id: str,
    parser: ReplayParser,
    min_rating: int = 0,
    max_age_days: int | None = None,
    max_replays: int | None = None,
) -> dict[str, int]:
    """Import replays from a Parquet file into the database."""
    stats = {"total": 0, "imported": 0, "skipped_exists": 0, "skipped_rating": 0,
             "skipped_age": 0, "parse_error": 0}

    age_cutoff = None
    if max_age_days:
        age_cutoff = int(time.time()) - max_age_days * 86400

    log.info("Reading parquet: %s", parquet_path.name)
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    stats["total"] = len(df)
    log.info("Loaded %d replays from %s", len(df), parquet_path.name)

    # Sort by uploadtime descending (newest first) for recency filtering
    if "uploadtime" in df.columns:
        df = df.sort_values("uploadtime", ascending=False)

    batch_count = 0
    for idx, row in df.iterrows():
        if max_replays and stats["imported"] >= max_replays:
            break

        replay_id = str(row.get("id", ""))
        if not replay_id:
            stats["parse_error"] += 1
            continue

        # Check age filter
        upload_time = row.get("uploadtime")
        if age_cutoff and upload_time and int(upload_time) < age_cutoff:
            stats["skipped_age"] += 1
            continue

        # Check if already exists
        if await db.replay_exists(replay_id):
            stats["skipped_exists"] += 1
            continue

        # Parse the log
        log_text = str(row.get("log", ""))
        if not log_text or len(log_text) < 50:
            stats["parse_error"] += 1
            continue

        try:
            battle = parser.parse(log_text, format_id)
        except Exception:
            stats["parse_error"] += 1
            continue

        if not battle.is_valid():
            stats["parse_error"] += 1
            continue

        # Check rating filter
        r1 = battle.rating1 or 0
        r2 = battle.rating2 or 0
        if min_rating > 0 and r1 < min_rating and r2 < min_rating:
            stats["skipped_rating"] += 1
            continue

        # Insert into database
        await db.insert_replay(
            replay_id=replay_id,
            format_id=format_id,
            player1=battle.player1,
            player2=battle.player2,
            winner=battle.winner,
            rating1=battle.rating1,
            rating2=battle.rating2,
            turns=battle.turns,
            upload_time=int(upload_time) if upload_time else None,
        )

        team1_data = [p.to_dict() for p in battle.team1]
        team2_data = [p.to_dict() for p in battle.team2]
        await db.insert_team(replay_id, 1, team1_data)
        await db.insert_team(replay_id, 2, team2_data)

        stats["imported"] += 1
        batch_count += 1

        # Commit in batches
        if batch_count >= 500:
            await db.commit()
            batch_count = 0
            log.info(
                "  Progress: imported=%d, skipped_exists=%d, skipped_rating=%d, "
                "skipped_age=%d, errors=%d",
                stats["imported"], stats["skipped_exists"], stats["skipped_rating"],
                stats["skipped_age"], stats["parse_error"],
            )

    await db.commit()
    return stats


async def main():
    arg_parser = argparse.ArgumentParser(description="Import HolidayOugi HF dataset")
    arg_parser.add_argument("--format", type=str, default=None, help="Only import this format")
    arg_parser.add_argument("--formats", type=str, default=None,
                            help="Comma-separated list of formats to import")
    arg_parser.add_argument("--gen", type=int, default=None,
                            help="Only import formats for this generation (1-9)")
    arg_parser.add_argument("--max-per-format", type=int, default=None,
                            help="Max replays to import per format")
    arg_parser.add_argument("--min-rating", type=int, default=0,
                            help="Only import games where at least one player >= this rating")
    arg_parser.add_argument("--max-age-days", type=int, default=None,
                            help="Only import replays from the last N days")
    arg_parser.add_argument("--skip-download", action="store_true",
                            help="Skip download, use existing parquet files")
    arg_parser.add_argument("--cleanup", action="store_true",
                            help="Delete parquet files after successful import (saves disk)")
    arg_parser.add_argument("--config", type=str, default=None)
    args = arg_parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    db_path = cfg.get("database", {}).get("path", "data/replays.db")

    # Determine which formats to import
    if args.formats:
        formats = [f.strip() for f in args.formats.split(",")]
    elif args.format:
        formats = [args.format]
    elif args.gen is not None:
        # Filter to only formats for the specified generation
        prefix = f"gen{args.gen}"
        formats = [fmt for fmt in FORMAT_TO_HF_FILES if fmt.startswith(prefix)]
        if not formats:
            log.error("No HuggingFace mappings found for gen %d", args.gen)
            return
        log.info("Importing %d formats for Gen %d: %s", len(formats), args.gen, formats)
    else:
        # Import all formats from config that have HF mappings
        formats = []
        for fmt_list in cfg.get("formats", {}).values():
            formats.extend(fmt_list)

    db = Database(db_path)
    await db.connect()
    parser = ReplayParser()

    total_stats = {"imported": 0, "skipped_exists": 0, "parse_error": 0}
    start_time = time.time()

    for fmt in formats:
        if fmt not in FORMAT_TO_HF_FILES:
            log.warning("No HuggingFace mapping for format: %s (skipping)", fmt)
            continue

        log.info("=" * 60)
        log.info("FORMAT: %s", fmt)
        log.info("=" * 60)

        existing_count = await db.get_replay_count(fmt)
        log.info("Existing replays in DB: %d", existing_count)

        # Discover and download parquet files
        for base_name in FORMAT_TO_HF_FILES[fmt]:
            downloaded_files = []

            if not args.skip_download:
                file_entries = discover_parquet_urls(base_name)
                if not file_entries:
                    log.warning("No parquet files found for %s", base_name)
                    continue
                log.info("Found %d parquet file(s) for %s", len(file_entries), base_name)

                for url, filename in file_entries:
                    dest = DOWNLOAD_DIR / filename
                    download_file(url, dest)
                    downloaded_files.append(dest)

            # Find downloaded files (escape glob special chars in bracket-heavy filenames)
            parquet_files = sorted(
                str(p) for p in DOWNLOAD_DIR.iterdir()
                if p.name.startswith(base_name) and p.name.endswith(".parquet")
            ) if DOWNLOAD_DIR.exists() else []

            if not parquet_files:
                log.warning("No parquet files found matching: %s", base_name)
                continue

            for pf in parquet_files:
                fmt_stats = await import_parquet(
                    db=db,
                    parquet_path=Path(pf),
                    format_id=fmt,
                    parser=parser,
                    min_rating=args.min_rating,
                    max_age_days=args.max_age_days,
                    max_replays=args.max_per_format,
                )
                log.info("  File %s: %s", Path(pf).name, fmt_stats)

                for k in total_stats:
                    total_stats[k] += fmt_stats.get(k, 0)

                # Cleanup: delete parquet after successful import
                if args.cleanup and fmt_stats["imported"] > 0:
                    try:
                        os.remove(pf)
                        log.info("  Cleaned up: %s", Path(pf).name)
                    except OSError as e:
                        log.warning("  Failed to cleanup %s: %s", Path(pf).name, e)

                # Check if we've hit the per-format limit
                if args.max_per_format and fmt_stats["imported"] >= args.max_per_format:
                    break

        new_count = await db.get_replay_count(fmt)
        log.info("Final count for %s: %d (added %d)", fmt, new_count, new_count - existing_count)

    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info("IMPORT COMPLETE in %.0f seconds", elapsed)
    log.info("Total imported: %d | Skipped (exists): %d | Parse errors: %d",
             total_stats["imported"], total_stats["skipped_exists"], total_stats["parse_error"])
    log.info("=" * 60)

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
