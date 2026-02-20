#!/usr/bin/env python3
"""Evaluate trained models and display metagame analysis.

Usage:
    python scripts/evaluate.py --format gen9ou --action test
    python scripts/evaluate.py --format gen9ou --action meta
    python scripts/evaluate.py --format gen9ou --action compare-team --team team.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from showdown.config import load_config
from showdown.data.database import Database
from showdown.data.pokemon_data import PokemonDataLoader
from showdown.data.features import FeatureExtractor
from showdown.data.preprocessor import DataPreprocessor
from showdown.models.win_predictor import WinPredictor
from showdown.models.xgb_predictor import XGBPredictor
from showdown.models.ensemble import EnsemblePredictor
from showdown.models.trainer import Trainer
from showdown.teambuilder.meta_analysis import MetaAnalyzer
from showdown.teambuilder.evaluator import TeamEvaluator
from showdown.utils.logging_config import setup_logging


async def main():
    parser = argparse.ArgumentParser(description="Evaluate models and analyze metagame")
    parser.add_argument("--format", type=str, required=True)
    parser.add_argument("--action", type=str, required=True,
                        choices=["test", "meta", "compare-team", "calibrate"])
    parser.add_argument("--team", type=str, help="Team JSON file for compare-team")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    log = setup_logging("INFO")
    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    checkpoint_dir = train_cfg.get("checkpoint_dir", "data/checkpoints")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    db_path = cfg.get("database", {}).get("path", "data/showdown.db")
    db = Database(db_path)
    await db.connect()

    pkmn_data = PokemonDataLoader()
    await pkmn_data.load()

    # Load vocab + feature extractor
    vocab_path = Path(checkpoint_dir) / f"vocab_{args.format}.json"
    if not vocab_path.exists():
        log.error("No vocabulary found. Train a model first.")
        await db.close()
        return

    with open(vocab_path) as f:
        vocab = json.load(f)

    fe = FeatureExtractor(pokemon_data=pkmn_data)
    fe._species_idx = vocab["species"]
    fe._move_idx = vocab["moves"]
    fe._item_idx = vocab["items"]
    fe._ability_idx = vocab["abilities"]

    # Load models
    trainer = Trainer(checkpoint_dir=checkpoint_dir, device=device)
    vocab_sizes = fe.vocab_sizes

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

    neural_loaded = trainer.load_neural(neural_model, args.format)
    if neural_loaded:
        neural_model.eval()

    xgb_model = XGBPredictor()
    xgb_loaded = trainer.load_xgboost(xgb_model, args.format)

    ensemble = EnsemblePredictor(
        neural_model=neural_model if neural_loaded else None,
        xgb_model=xgb_model if xgb_loaded else None,
        feature_extractor=fe,
        device=device,
    )

    if args.action == "test":
        await _run_test(db, fe, neural_model if neural_loaded else None,
                        xgb_model if xgb_loaded else None,
                        args.format, train_cfg, log)

    elif args.action == "meta":
        await _show_meta(db, args.format, log)

    elif args.action == "compare-team":
        if not args.team:
            log.error("--team required for compare-team action")
        else:
            await _compare_team(db, ensemble, args.format, args.team, log)

    elif args.action == "calibrate":
        await _calibrate(db, ensemble, fe, args.format, train_cfg, log)

    await db.close()


async def _run_test(db, fe, neural_model, xgb_model, format_id, train_cfg, log):
    """Run full test set evaluation."""
    min_rating = train_cfg.get("min_rating_filter", 1500)
    battles = await db.get_training_battles(format_id, min_rating=min_rating)

    if not battles:
        log.error("No battles found")
        return

    preprocessor = DataPreprocessor(feature_extractor=fe)

    if neural_model is not None:
        log.info("--- Neural Network Test Results ---")
        neural_data = preprocessor.prepare_neural_dataset(battles, augment=False)
        test_samples = neural_data["test"]
        if test_samples:
            correct = 0
            total = 0
            for sample in test_samples:
                inputs = {
                    k: torch.tensor(v).unsqueeze(0)
                    for k, v in sample.items() if k != "label"
                }
                with torch.no_grad():
                    pred = neural_model(**inputs).item()
                predicted = 1 if pred > 0.5 else 0
                actual = int(sample["label"])
                if predicted == actual:
                    correct += 1
                total += 1
            log.info("Accuracy: %.4f (%d/%d)", correct / total, correct, total)

    if xgb_model is not None:
        log.info("--- XGBoost Test Results ---")
        xgb_data = preprocessor.prepare_xgboost_dataset(battles, augment=False)
        X_test, y_test = xgb_data["test"]
        if len(X_test) > 0:
            results = xgb_model.evaluate(X_test, y_test)
            log.info("Accuracy: %.4f", results["accuracy"])
            log.info("AUC: %.4f", results["auc"])
            log.info("LogLoss: %.4f", results["logloss"])


async def _show_meta(db, format_id, log):
    """Display metagame summary."""
    analyzer = MetaAnalyzer(db)
    summary = await analyzer.get_meta_summary(format_id, top_n=30)

    if "error" in summary:
        log.error(summary["error"])
        return

    log.info("=== Metagame Summary: %s (%s) ===", format_id, summary.get("year_month", "?"))
    log.info("Total Pokemon in meta: %d", summary.get("total_pokemon", 0))
    log.info("")

    for i, pkmn in enumerate(summary.get("pokemon", []), 1):
        log.info(
            "%2d. %-20s  %.2f%%  | Moves: %s | Items: %s",
            i,
            pkmn["species"],
            pkmn["usage_pct"],
            ", ".join(list(pkmn.get("moves", {}).keys())[:4]),
            ", ".join(list(pkmn.get("items", {}).keys())[:2]),
        )


async def _compare_team(db, ensemble, format_id, team_path, log):
    """Evaluate a specific team against the meta."""
    with open(team_path) as f:
        team = json.load(f)

    analyzer = MetaAnalyzer(db)
    meta_teams = await analyzer.build_meta_teams(format_id, n_teams=50)
    evaluator = TeamEvaluator(predictor=ensemble, meta_teams=meta_teams)

    detailed = evaluator.evaluate_detailed(team)
    log.info("=== Team Evaluation: %s ===", format_id)
    log.info("Overall predicted winrate: %.1f%%", detailed["overall_winrate"] * 100)
    log.info("Matchups above 50%%: %d/%d",
             detailed["matchups_above_50"], detailed["total_matchups"])

    log.info("\nWorst matchups:")
    for m in detailed["worst_matchups"]:
        log.info("  %.1f%% vs %s", m["win_prob"] * 100, ", ".join(m["opponent"][:3]))

    log.info("\nBest matchups:")
    for m in detailed["best_matchups"]:
        log.info("  %.1f%% vs %s", m["win_prob"] * 100, ", ".join(m["opponent"][:3]))


async def _calibrate(db, ensemble, fe, format_id, train_cfg, log):
    """Calibrate ensemble weights on validation data."""
    min_rating = train_cfg.get("min_rating_filter", 1500)
    battles = await db.get_training_battles(format_id, min_rating=min_rating, limit=1000)

    if len(battles) < 100:
        log.error("Not enough battles for calibration")
        return

    # Use last 20% as calibration set
    cal_battles = battles[int(len(battles) * 0.8):]
    labels = [1.0 if b["winner"] == 1 else 0.0 for b in cal_battles]

    w_neural, w_xgb = ensemble.calibrate_weights(cal_battles, labels)
    log.info("Calibrated weights: neural=%.2f, xgb=%.2f", w_neural, w_xgb)


if __name__ == "__main__":
    asyncio.run(main())
