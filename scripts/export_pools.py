#!/usr/bin/env python3
"""Export meta teams and Pokemon pools to JSON for offline/packaged use.

Runs against the live database and dumps one JSON file per loaded format
into data/pools/{format}.json. The server can load these instead of
querying the database, enabling the packaged .exe to work without the
22GB replays.db.

Usage:
    python scripts/export_pools.py
    python scripts/export_pools.py --formats gen9ou gen9uu
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown.config import load_config
from showdown.data.database import Database
from showdown.data.pokemon_data import PokemonDataLoader
from showdown.teambuilder.meta_analysis import MetaAnalyzer
from showdown.utils.constants import extract_gen

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("export_pools")


async def export_format(
    db: Database,
    pkmn_data: PokemonDataLoader,
    fmt: str,
    output_dir: Path,
) -> bool:
    """Export meta teams + pokemon pool for a single format."""
    gen = extract_gen(fmt)
    meta_analyzer = MetaAnalyzer(db, pokemon_data=pkmn_data, gen=gen)

    # Build meta teams (same logic as state.py)
    meta_teams = await meta_analyzer.build_meta_teams(fmt, n_teams=50)
    if not meta_teams:
        log.warning("No meta teams for %s, skipping", fmt)
        return False

    # Build pokemon pool (try usage stats first, fall back to replays)
    pokemon_pool = []
    year_month = await db.get_latest_usage_month(fmt)
    if year_month:
        for rating in [1825, 1760, 1630, 1500, 0]:
            raw_usage = await db.get_usage_stats(fmt, year_month, rating)
            if raw_usage:
                from showdown.scraper.stats_scraper import StatsScraper
                usage_parsed = StatsScraper(db).parse_usage_data(raw_usage)
                pokemon_pool = meta_analyzer.build_full_pokemon_pool(
                    usage_parsed, top_n=80, sets_per_pokemon=4
                )
                break

    if not pokemon_pool:
        log.info("No usage stats for %s, building pool from replays...", fmt)
        pokemon_pool = await meta_analyzer.build_pool_from_replays(
            fmt, top_n=80, sets_per_pokemon=4
        )

    if not pokemon_pool:
        log.warning("No pokemon pool for %s, skipping", fmt)
        return False

    # Validate moves against gen + learnability
    import re as _re
    def _to_id(name: str) -> str:
        return _re.sub(r"[^a-z0-9]", "", name.lower())

    validated_pool = []
    stripped_moves = 0
    dropped_sets = 0
    for pset in pokemon_pool:
        species_id = _to_id(pset.get("species", ""))
        moves = pset.get("moves", [])
        valid_moves = []
        for m in moves:
            mid = _to_id(m)
            if pkmn_data.move_exists_in_gen(mid, gen) and pkmn_data.can_learn_move(species_id, mid, gen):
                valid_moves.append(m)
            else:
                stripped_moves += 1
        if len(valid_moves) >= 2:
            pset = dict(pset)
            pset["moves"] = valid_moves
            # Strip gen-inappropriate fields
            if gen < 3:
                pset.pop("ability", None)
                pset.pop("nature", None)
                pset.pop("evs", None)
                pset.pop("ivs", None)
            if gen < 2:
                pset.pop("item", None)
            if gen < 9:
                pset.pop("tera_type", None)
            validated_pool.append(pset)
        else:
            dropped_sets += 1
    if stripped_moves or dropped_sets:
        log.info("Gen %d validation for %s: stripped %d illegal moves, dropped %d sets (%d -> %d)",
                 gen, fmt, stripped_moves, dropped_sets, len(pokemon_pool), len(validated_pool))
    pokemon_pool = validated_pool

    # Same validation for meta teams
    validated_teams = []
    for team in meta_teams:
        valid_team = []
        for pset in team:
            species_id = _to_id(pset.get("species", ""))
            moves = pset.get("moves", [])
            valid_moves = [m for m in moves
                           if pkmn_data.move_exists_in_gen(_to_id(m), gen)
                           and pkmn_data.can_learn_move(species_id, _to_id(m), gen)]
            pset = dict(pset)
            pset["moves"] = valid_moves if valid_moves else moves[:1]
            if gen < 3:
                pset.pop("ability", None)
                pset.pop("nature", None)
                pset.pop("evs", None)
                pset.pop("ivs", None)
            if gen < 2:
                pset.pop("item", None)
            if gen < 9:
                pset.pop("tera_type", None)
            valid_team.append(pset)
        validated_teams.append(valid_team)
    meta_teams = validated_teams

    # Write JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{fmt}.json"
    data = {
        "format_id": fmt,
        "meta_teams": meta_teams,
        "pokemon_pool": pokemon_pool,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_kb = out_path.stat().st_size / 1024
    log.info(
        "Exported %s: %d meta teams, %d pool sets (%.1f KB)",
        fmt, len(meta_teams), len(pokemon_pool), size_kb,
    )
    return True


async def main():
    parser = argparse.ArgumentParser(description="Export meta pools to JSON")
    parser.add_argument("--formats", nargs="*", help="Specific formats to export (default: all with vocab)")
    parser.add_argument("--config", default=None, help="Config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg.get("database", {}).get("path", "data/replays.db")
    checkpoint_dir = cfg.get("training", {}).get("checkpoint_dir", "data/checkpoints")

    db = Database(db_path)
    await db.connect()

    pkmn_data = PokemonDataLoader()
    await pkmn_data.load()

    # Determine which formats to export
    if args.formats:
        formats_to_export = args.formats
    else:
        # Export all formats that have a trained model (vocab file exists)
        formats_to_export = []
        for fmt_list in cfg.get("formats", {}).values():
            for fmt in fmt_list:
                vocab_path = Path(checkpoint_dir) / f"vocab_{fmt}.json"
                if vocab_path.exists():
                    formats_to_export.append(fmt)

    output_dir = Path("data/pools")
    log.info("Exporting %d formats to %s", len(formats_to_export), output_dir)

    exported = 0
    for fmt in formats_to_export:
        try:
            if await export_format(db, pkmn_data, fmt, output_dir):
                exported += 1
        except Exception as e:
            log.error("Failed to export %s: %s", fmt, e)

    await db.close()
    log.info("Done: exported %d / %d formats", exported, len(formats_to_export))


if __name__ == "__main__":
    asyncio.run(main())
