#!/usr/bin/env python3
"""Build optimized teams using the genetic algorithm + trained models.

Usage:
    python scripts/build_team.py --format gen9ou --results 5
    python scripts/build_team.py --format gen9vgc2024regg --results 3
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from showdown.config import load_config
from showdown.data.database import Database
from showdown.data.pokemon_data import PokemonDataLoader
from showdown.data.features import FeatureExtractor
from showdown.models.win_predictor import WinPredictor
from showdown.models.xgb_predictor import XGBPredictor
from showdown.models.ensemble import EnsemblePredictor
from showdown.models.trainer import Trainer
from showdown.teambuilder.constraints import FormatConstraints
from showdown.teambuilder.evaluator import TeamEvaluator
from showdown.teambuilder.genetic import GeneticTeamBuilder
from showdown.teambuilder.meta_analysis import MetaAnalyzer
from showdown.utils.logging_config import setup_logging


async def main():
    parser = argparse.ArgumentParser(description="Build optimized Pokemon teams")
    parser.add_argument("--format", type=str, required=True, help="Format (e.g. gen9ou)")
    parser.add_argument("--results", type=int, default=5, help="Number of teams to output")
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--meta-teams", type=int, default=None, help="Number of meta teams")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    log = setup_logging("INFO")
    cfg = load_config(args.config)
    tb_cfg = cfg.get("teambuilder", {})
    model_cfg = cfg.get("model", {})
    checkpoint_dir = cfg.get("training", {}).get("checkpoint_dir", "data/checkpoints")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Connect to database
    db_path = cfg.get("database", {}).get("path", "data/showdown.db")
    db = Database(db_path)
    await db.connect()

    # Load Pokemon data
    log.info("Loading Pokemon data...")
    pkmn_data = PokemonDataLoader()
    await pkmn_data.load()

    # Load vocabulary
    vocab_path = Path(checkpoint_dir) / f"vocab_{args.format}.json"
    if not vocab_path.exists():
        log.error("No vocabulary found at %s. Train a model first.", vocab_path)
        await db.close()
        return

    with open(vocab_path) as f:
        vocab = json.load(f)

    # Build feature extractor with saved vocab
    fe = FeatureExtractor(pokemon_data=pkmn_data)
    fe._species_idx = vocab["species"]
    fe._move_idx = vocab["moves"]
    fe._item_idx = vocab["items"]
    fe._ability_idx = vocab["abilities"]

    # Load models
    trainer = Trainer(checkpoint_dir=checkpoint_dir, device=device)

    neural_model = None
    xgb_model = None

    # Try loading neural model
    vocab_sizes = fe.vocab_sizes
    try:
        neural_model = WinPredictor(
            num_species=vocab_sizes["species"],
            num_moves=vocab_sizes["moves"],
            num_items=vocab_sizes["items"],
            num_abilities=vocab_sizes["abilities"],
            species_dim=model_cfg.get("pokemon_embed_dim", 64),
            move_dim=model_cfg.get("move_embed_dim", 32),
            item_dim=model_cfg.get("item_embed_dim", 32),
            ability_dim=model_cfg.get("ability_embed_dim", 32),
            pokemon_dim=model_cfg.get("pokemon_hidden_dim", 128),
            team_dim=model_cfg.get("team_hidden_dim", 256),
            attention_heads=model_cfg.get("attention_heads", 4),
            dropout=model_cfg.get("dropout", 0.2),
        )
        loaded = trainer.load_neural(neural_model, args.format)
        if loaded is None:
            neural_model = None
            log.warning("No neural model checkpoint found")
        else:
            neural_model.eval()
            log.info("Neural model loaded")
    except Exception as e:
        log.warning("Failed to load neural model: %s", e)
        neural_model = None

    # Try loading XGBoost model
    try:
        xgb_model = XGBPredictor()
        if not trainer.load_xgboost(xgb_model, args.format):
            xgb_model = None
            log.warning("No XGBoost model checkpoint found")
        else:
            log.info("XGBoost model loaded")
    except Exception as e:
        log.warning("Failed to load XGBoost model: %s", e)
        xgb_model = None

    if neural_model is None and xgb_model is None:
        log.error("No trained models found. Run train_model.py first.")
        await db.close()
        return

    # Create ensemble predictor
    ensemble = EnsemblePredictor(
        neural_model=neural_model,
        xgb_model=xgb_model,
        feature_extractor=fe,
        device=device,
    )

    # Build meta teams for evaluation
    log.info("Building meta team distribution...")
    meta_analyzer = MetaAnalyzer(db)
    n_meta = args.meta_teams or tb_cfg.get("meta_sample_size", 50)
    meta_teams = await meta_analyzer.build_meta_teams(
        args.format, n_teams=n_meta
    )
    log.info("Built %d meta teams for evaluation", len(meta_teams))

    if not meta_teams:
        log.error("No meta teams available. Scrape usage stats first.")
        await db.close()
        return

    # Build rich pokemon pool from usage stats (multiple sets per Pokemon)
    log.info("Building Pokemon pool from usage data...")
    year_month = await db.get_latest_usage_month(args.format)
    raw_usage = None
    if year_month:
        for rating in [1825, 1760, 1630, 1500, 0]:
            raw_usage = await db.get_usage_stats(args.format, year_month, rating)
            if raw_usage:
                log.info("Using usage stats at rating threshold %d", rating)
                break
    if raw_usage:
        from showdown.scraper.stats_scraper import StatsScraper
        usage_parsed = StatsScraper(db).parse_usage_data(raw_usage)
        pokemon_pool = meta_analyzer.build_full_pokemon_pool(
            usage_parsed, top_n=80, sets_per_pokemon=4
        )
    else:
        pokemon_pool = _build_pool_from_meta(meta_teams, pkmn_data)
    log.info("Pokemon pool: %d sets across %d species",
             len(pokemon_pool),
             len(set(p["species"] for p in pokemon_pool)))

    # Setup constraints
    constraints = FormatConstraints(args.format, pokemon_data=pkmn_data)

    # Setup evaluator
    evaluator = TeamEvaluator(predictor=ensemble, meta_teams=meta_teams)

    # Run genetic algorithm
    log.info("=" * 60)
    log.info("STARTING GENETIC TEAM BUILDER")
    log.info("Format: %s | Population: %d | Generations: %d",
             args.format,
             args.population or tb_cfg.get("population_size", 200),
             args.generations or tb_cfg.get("generations", 500))
    log.info("=" * 60)

    builder = GeneticTeamBuilder(
        evaluator=evaluator,
        constraints=constraints,
        pokemon_pool=pokemon_pool,
        population_size=args.population or tb_cfg.get("population_size", 200),
        generations=args.generations or tb_cfg.get("generations", 500),
        mutation_rate=tb_cfg.get("mutation_rate", 0.15),
        crossover_rate=tb_cfg.get("crossover_rate", 0.7),
        elite_size=tb_cfg.get("elite_size", 20),
        tournament_size=tb_cfg.get("tournament_size", 5),
    )

    results = builder.build(n_results=args.results)

    # Display results
    log.info("=" * 60)
    log.info("TEAM BUILDER RESULTS")
    log.info("=" * 60)

    output_data = []
    for i, result in enumerate(results, 1):
        team = result["team"]
        fitness = result["fitness"]
        gen = result["generation"]

        log.info("\n--- Team #%d (predicted winrate: %.1f%%, found at gen %d) ---",
                 i, fitness * 100, gen)

        team_output = {
            "rank": i,
            "predicted_winrate": round(fitness * 100, 2),
            "generation_found": gen,
            "pokemon": [],
        }

        for pkmn in team:
            species = pkmn.get("species", "?")
            ability = pkmn.get("ability", "?")
            item = pkmn.get("item", "?")
            moves = pkmn.get("moves", [])
            tera = pkmn.get("tera_type", "?")

            log.info("  %s @ %s | %s | Tera: %s", species, item, ability, tera)
            log.info("    Moves: %s", ", ".join(moves))

            team_output["pokemon"].append({
                "species": species,
                "ability": ability,
                "item": item,
                "moves": moves,
                "tera_type": tera,
            })

        # Detailed evaluation
        detailed = evaluator.evaluate_detailed(team)
        log.info("  Matchups above 50%%: %d/%d",
                 detailed["matchups_above_50"], detailed["total_matchups"])

        team_output["evaluation"] = {
            "matchups_above_50": detailed["matchups_above_50"],
            "total_matchups": detailed["total_matchups"],
        }
        output_data.append(team_output)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(checkpoint_dir) / f"teams_{args.format}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log.info("\nResults saved to %s", output_path)

    await db.close()


def _build_pool_from_meta(
    meta_teams: list[list[dict]],
    pkmn_data: PokemonDataLoader,
) -> list[dict]:
    """Build a pool of Pokemon sets from meta teams + usage data.

    Each unique (species, moveset, item, ability) combination becomes a pool entry.
    """
    seen = set()
    pool = []

    for team in meta_teams:
        for pkmn in team:
            species = pkmn.get("species", "")
            moves = tuple(sorted(pkmn.get("moves", [])))
            item = pkmn.get("item", "")
            ability = pkmn.get("ability", "")

            key = (species, moves, item, ability)
            if key not in seen and species:
                seen.add(key)
                pool.append(pkmn)

    return pool


if __name__ == "__main__":
    asyncio.run(main())
