#!/usr/bin/env python3
"""Calibration analysis for win prediction models.

Generates reliability diagrams: predicted probability vs actual win rate by decile.
Computes Expected Calibration Error (ECE) and Brier score.

Usage:
    python scripts/calibration.py --format gen9ou
    python scripts/calibration.py --format gen9ou --model xgboost
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown.config import load_config
from showdown.data.database import Database
from showdown.data.pokemon_data import PokemonDataLoader
from showdown.data.features import FeatureExtractor
from showdown.data.preprocessor import DataPreprocessor
from showdown.models.win_predictor import WinPredictor
from showdown.models.xgb_predictor import XGBPredictor
from showdown.models.trainer import Trainer
from showdown.utils.logging_config import setup_logging


def _list_to_arrays(samples: list[dict]) -> dict[str, np.ndarray]:
    if not samples:
        return {}
    result = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        if isinstance(vals[0], np.ndarray):
            result[key] = np.stack(vals)
        else:
            result[key] = np.array(vals)
    return result


def calibration_curve(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10):
    """Compute calibration curve data.

    Returns:
        bin_centers: center of each probability bin
        bin_true_freq: actual win rate in each bin
        bin_pred_mean: mean predicted probability in each bin
        bin_counts: number of samples in each bin
        ece: Expected Calibration Error
        brier: Brier score
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_true_freq = []
    bin_pred_mean = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)

        count = mask.sum()
        bin_counts.append(int(count))
        if count > 0:
            bin_centers.append((lo + hi) / 2)
            bin_true_freq.append(float(y_true[mask].mean()))
            bin_pred_mean.append(float(y_pred[mask].mean()))
        else:
            bin_centers.append((lo + hi) / 2)
            bin_true_freq.append(None)
            bin_pred_mean.append(None)

    # ECE: weighted average of |predicted - actual| per bin
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        if bin_counts[i] > 0 and bin_true_freq[i] is not None:
            ece += (bin_counts[i] / total) * abs(bin_pred_mean[i] - bin_true_freq[i])

    brier = float(np.mean((y_pred - y_true) ** 2))

    return bin_centers, bin_true_freq, bin_pred_mean, bin_counts, ece, brier


def print_calibration_table(
    name: str,
    bin_centers: list,
    bin_true_freq: list,
    bin_pred_mean: list,
    bin_counts: list,
    ece: float,
    brier: float,
    log,
):
    """Print ASCII reliability diagram."""
    log.info("")
    log.info("=== %s Calibration ===", name)
    log.info("")
    log.info("%-12s  %-12s  %-12s  %-8s  %-30s", "Pred Range", "Actual WR", "Mean Pred", "Count", "Diagram")
    log.info("-" * 80)

    for i, (center, true_f, pred_m, count) in enumerate(
        zip(bin_centers, bin_true_freq, bin_pred_mean, bin_counts)
    ):
        lo = center - 0.05
        hi = center + 0.05
        range_str = f"[{lo:.2f}, {hi:.2f})"

        if true_f is None or count == 0:
            log.info("%-12s  %-12s  %-12s  %-8d  (empty)", range_str, "-", "-", count)
            continue

        # ASCII bar: | for predicted, * for actual
        bar_width = 40
        pred_bar = int(pred_m * bar_width)
        true_bar = int(true_f * bar_width)

        bar = list("." * bar_width)
        # Mark predicted with |
        if 0 <= pred_bar < bar_width:
            bar[pred_bar] = "|"
        # Mark actual with *
        if 0 <= true_bar < bar_width:
            bar[true_bar] = "*"
        # If they overlap, use #
        if pred_bar == true_bar and 0 <= pred_bar < bar_width:
            bar[pred_bar] = "#"

        bar_str = "".join(bar)
        log.info(
            "%-12s  %-12.4f  %-12.4f  %-8d  %s",
            range_str, true_f, pred_m, count, bar_str,
        )

    log.info("-" * 80)
    log.info("ECE (Expected Calibration Error): %.4f", ece)
    log.info("Brier Score: %.4f", brier)
    log.info("Legend: | = predicted, * = actual, # = overlap (perfectly calibrated)")
    log.info("")


async def main():
    parser = argparse.ArgumentParser(description="Model Calibration Analysis")
    parser.add_argument("--format", type=str, required=True)
    parser.add_argument("--model", type=str, default="both", choices=["neural", "xgboost", "both"])
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    log = setup_logging("INFO")
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    checkpoint_dir = train_cfg.get("checkpoint_dir", "data/checkpoints")

    min_rating = args.min_rating or train_cfg.get("min_rating_filter", 1500)

    # Load data
    db_path = cfg.get("database", {}).get("path", "data/replays.db")
    db = Database(db_path)
    await db.connect()

    log.info("Loading battles for %s...", args.format)
    battles = await db.get_training_battles(args.format, min_rating=min_rating, limit=args.limit)
    log.info("Loaded %d battles", len(battles))

    if len(battles) < 100:
        log.error("Not enough battles (%d).", len(battles))
        await db.close()
        return

    # Load Pokemon data + vocab
    pkmn_data = PokemonDataLoader()
    await pkmn_data.load()

    vocab_path = Path(checkpoint_dir) / f"vocab_{args.format}.json"
    if not vocab_path.exists():
        log.error("No vocab file at %s. Train models first.", vocab_path)
        await db.close()
        return

    with open(vocab_path) as f:
        vocab = json.load(f)

    fe = FeatureExtractor(pokemon_data=pkmn_data)
    fe._species_idx = vocab["species"]
    fe._move_idx = vocab["moves"]
    fe._item_idx = vocab["items"]
    fe._ability_idx = vocab["abilities"]

    # Use same split as training (test set only)
    preprocessor = DataPreprocessor(
        feature_extractor=fe,
        train_split=train_cfg.get("train_split", 0.8),
        val_split=train_cfg.get("val_split", 0.1),
        test_split=train_cfg.get("test_split", 0.1),
    )

    results = {"format": args.format, "n_battles": len(battles)}

    # ---- Neural calibration ----
    if args.model in ("neural", "both"):
        import torch

        neural_data = preprocessor.prepare_neural_dataset(battles, augment=False)
        test_arrays = _list_to_arrays(neural_data["test"])

        if test_arrays:
            vocab_sizes = fe.vocab_sizes
            model = WinPredictor(
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
                dropout=model_cfg.get("dropout", 0.25),
                continuous_dim=model_cfg.get("continuous_dim", 42),
                rating_dim=model_cfg.get("rating_dim", 6),
            )

            trainer = Trainer(checkpoint_dir=checkpoint_dir)
            loaded = trainer.load_neural(model, args.format)
            if loaded:
                model.eval()
                test_loader = trainer._make_dataloader(test_arrays, batch_size=512, shuffle=False)
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        inputs, labels = trainer._unpack_batch(batch)
                        preds = model(**inputs)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                preds_arr = np.array(all_preds)
                labels_arr = np.array(all_labels)

                centers, true_freq, pred_mean, counts, ece, brier = calibration_curve(
                    labels_arr, preds_arr, n_bins=args.bins
                )
                print_calibration_table("Neural", centers, true_freq, pred_mean, counts, ece, brier, log)

                results["neural"] = {
                    "ece": float(ece),
                    "brier": float(brier),
                    "auc": float(roc_auc_score(labels_arr, preds_arr)),
                    "bins": [
                        {"center": c, "actual": t, "predicted": p, "count": n}
                        for c, t, p, n in zip(centers, true_freq, pred_mean, counts)
                    ],
                }
            else:
                log.warning("No neural checkpoint found for %s", args.format)

    # ---- XGBoost calibration ----
    if args.model in ("xgboost", "both"):
        xgb_data = preprocessor.prepare_xgboost_dataset(battles, augment=False)
        X_test, y_test = xgb_data["test"]

        if len(X_test) > 0:
            xgb_model = XGBPredictor()
            trainer = Trainer(checkpoint_dir=checkpoint_dir)
            if trainer.load_xgboost(xgb_model, args.format):
                preds_arr = xgb_model.predict(X_test)
                labels_arr = y_test

                centers, true_freq, pred_mean, counts, ece, brier = calibration_curve(
                    labels_arr, preds_arr, n_bins=args.bins
                )
                print_calibration_table("XGBoost", centers, true_freq, pred_mean, counts, ece, brier, log)

                results["xgboost"] = {
                    "ece": float(ece),
                    "brier": float(brier),
                    "auc": float(roc_auc_score(labels_arr, preds_arr)),
                    "bins": [
                        {"center": c, "actual": t, "predicted": p, "count": n}
                        for c, t, p, n in zip(centers, true_freq, pred_mean, counts)
                    ],
                }
            else:
                log.warning("No XGBoost checkpoint found for %s", args.format)

    # Save results
    out_path = Path("data") / f"calibration_{args.format}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_path)

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
