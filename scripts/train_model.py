#!/usr/bin/env python3
"""Train win prediction models.

Usage:
    python scripts/train_model.py --format gen9ou --model neural
    python scripts/train_model.py --format gen9ou --model xgboost
    python scripts/train_model.py --format gen9ou --model both
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown.config import load_config
from showdown.data.database import Database
from showdown.data.pokemon_data import PokemonDataLoader
from showdown.data.features import FeatureExtractor
from showdown.utils.constants import extract_gen
from showdown.data.preprocessor import DataPreprocessor
from showdown.models.win_predictor import WinPredictor
from showdown.models.xgb_predictor import XGBPredictor
from showdown.models.ensemble import EnsemblePredictor
from showdown.models.trainer import Trainer
from showdown.utils.logging_config import setup_logging


async def main():
    parser = argparse.ArgumentParser(description="Train win prediction models")
    parser.add_argument("--format", type=str, required=True, help="Format (e.g. gen9ou)")
    parser.add_argument("--model", type=str, default="both",
                        choices=["neural", "xgboost", "both"],
                        help="Model type to train")
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max battles to use")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--max-age-days", type=int, default=None,
                        help="Only use replays from the last N days")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    log = setup_logging("INFO")
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    xgb_cfg = cfg.get("xgboost", {})

    min_rating = args.min_rating or train_cfg.get("min_rating_filter", 1500)
    max_age_days = args.max_age_days or train_cfg.get("max_replay_age_days", None)
    epochs = args.epochs or train_cfg.get("epochs", 100)
    batch_size = args.batch_size or train_cfg.get("batch_size", 256)
    lr = args.lr or train_cfg.get("learning_rate", 0.001)
    checkpoint_dir = train_cfg.get("checkpoint_dir", "data/checkpoints")

    # Load data
    db_path = cfg.get("database", {}).get("path", "data/showdown.db")
    db = Database(db_path)
    await db.connect()

    age_msg = f", max_age={max_age_days}d" if max_age_days else ""
    log.info("Loading battle data for %s (min_rating=%d%s)...", args.format, min_rating, age_msg)
    battles = await db.get_training_battles(
        args.format, min_rating=min_rating, limit=args.limit, max_age_days=max_age_days
    )
    log.info("Loaded %d battles", len(battles))

    if len(battles) < 100:
        log.error("Not enough battles to train (%d). Scrape more replays first.", len(battles))
        await db.close()
        return

    # Load Pokemon data for feature engineering
    log.info("Loading Pokemon data from Showdown repo...")
    pkmn_data = PokemonDataLoader()
    await pkmn_data.load()

    # Feature extraction
    gen = extract_gen(args.format)
    fe = FeatureExtractor(pokemon_data=pkmn_data, gen=gen)
    vocab = fe.build_vocab_from_battles(battles)
    vocab_sizes = fe.vocab_sizes
    log.info("Vocab sizes: %s", vocab_sizes)

    # Preprocessing
    preprocessor = DataPreprocessor(
        feature_extractor=fe,
        train_split=train_cfg.get("train_split", 0.8),
        val_split=train_cfg.get("val_split", 0.1),
        test_split=train_cfg.get("test_split", 0.1),
    )

    # Save vocab for inference
    vocab_path = Path(checkpoint_dir) / f"vocab_{args.format}.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    log.info("Saved vocabulary to %s", vocab_path)

    trainer = Trainer(checkpoint_dir=checkpoint_dir, device=args.device)

    # ---- Train neural model ----
    if args.model in ("neural", "both"):
        log.info("=" * 60)
        log.info("TRAINING NEURAL MODEL")
        log.info("=" * 60)

        neural_data = preprocessor.prepare_neural_dataset(battles, augment=True)

        # Convert list-of-dicts to dict-of-arrays
        train_arrays = _list_to_arrays(neural_data["train"])
        val_arrays = _list_to_arrays(neural_data["val"])
        test_arrays = _list_to_arrays(neural_data["test"])

        log.info(
            "Dataset sizes: train=%d, val=%d, test=%d",
            len(neural_data["train"]), len(neural_data["val"]), len(neural_data["test"]),
        )

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
            continuous_dim=model_cfg.get("continuous_dim", 64),
            rating_dim=model_cfg.get("rating_dim", 6),
        )

        neural_metrics = trainer.train_neural(
            model=model,
            train_data=train_arrays,
            val_data=val_arrays,
            format_id=args.format,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=train_cfg.get("patience", 10),
            db=db,
        )

        log.info("Neural model results: val_AUC=%.4f, best_epoch=%d",
                 neural_metrics["best_val_auc"], neural_metrics["best_epoch"])

        # Test set evaluation (with rating features as-is)
        test_loader_data = _list_to_arrays(neural_data["test"])
        if test_loader_data:
            from sklearn.metrics import roc_auc_score, accuracy_score
            import torch

            def _to_tensor(arr):
                t = torch.tensor(arr)
                if t.is_floating_point():
                    t = t.float()
                return t

            test_keys = [
                "team1_species", "team1_moves", "team1_items", "team1_abilities",
                "team2_species", "team2_moves", "team2_items", "team2_abilities",
                "rating_features", "label",
            ]
            if "team1_continuous" in test_loader_data:
                test_keys += ["team1_continuous", "team2_continuous"]

            test_dataset = torch.utils.data.TensorDataset(
                *[_to_tensor(test_loader_data[k]) for k in test_keys]
            )
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_metrics = trainer._evaluate_neural(model, test_loader, torch.nn.BCELoss())
            log.info("Neural TEST results (with ratings): AUC=%.4f, acc=%.4f",
                     test_metrics["auc"], test_metrics["accuracy"])

            # Team-only evaluation: equalize rating features to 1500/1500
            test_data_teamonly = {k: v.copy() if isinstance(v, np.ndarray) else v
                                 for k, v in test_loader_data.items()}
            equal_ratings = np.full_like(test_data_teamonly["rating_features"], 0.0)
            equal_ratings[:, 0] = 0.75  # r1/2000 = 1500/2000
            equal_ratings[:, 1] = 0.75  # r2/2000 = 1500/2000
            equal_ratings[:, 2] = 0.0   # diff = 0
            equal_ratings[:, 3] = 1.0   # has_both = True
            equal_ratings[:, 4] = 0.75  # max/2000 = 1500/2000
            equal_ratings[:, 5] = 0.0   # abs_diff = 0
            test_data_teamonly["rating_features"] = equal_ratings
            teamonly_dataset = torch.utils.data.TensorDataset(
                *[_to_tensor(test_data_teamonly[k]) for k in test_keys]
            )
            teamonly_loader = torch.utils.data.DataLoader(teamonly_dataset, batch_size=batch_size, shuffle=False)
            teamonly_metrics = trainer._evaluate_neural(model, teamonly_loader, torch.nn.BCELoss())
            log.info("Neural TEST results (TEAM-ONLY, no ratings): AUC=%.4f, acc=%.4f",
                     teamonly_metrics["auc"], teamonly_metrics["accuracy"])

    # ---- Train XGBoost model ----
    if args.model in ("xgboost", "both"):
        log.info("=" * 60)
        log.info("TRAINING XGBOOST MODEL")
        log.info("=" * 60)

        xgb_data = preprocessor.prepare_xgboost_dataset(battles, augment=True)
        X_train, y_train, w_train = xgb_data["train"]
        X_val, y_val, w_val = xgb_data["val"]
        X_test, y_test, w_test = xgb_data["test"]

        log.info(
            "XGBoost dataset: train=%d, val=%d, test=%d, features=%d",
            len(X_train), len(X_val), len(X_test), X_train.shape[1],
        )

        xgb_model = XGBPredictor(
            n_estimators=xgb_cfg.get("n_estimators", 2000),
            max_depth=xgb_cfg.get("max_depth", 4),
            learning_rate=xgb_cfg.get("learning_rate", 0.03),
            subsample=xgb_cfg.get("subsample", 0.7),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.5),
            colsample_bylevel=xgb_cfg.get("colsample_bylevel", 0.7),
            min_child_weight=xgb_cfg.get("min_child_weight", 5),
            gamma=xgb_cfg.get("gamma", 0.1),
            reg_alpha=xgb_cfg.get("reg_alpha", 0.5),
            reg_lambda=xgb_cfg.get("reg_lambda", 2.0),
            early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 80),
        )

        xgb_metrics = trainer.train_xgboost(
            xgb_model=xgb_model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            format_id=args.format,
            db=db,
            sample_weight=w_train,
        )

        # Test set evaluation (with rating features as-is)
        test_results = xgb_model.evaluate(X_test, y_test)
        log.info("XGBoost TEST results (with ratings): accuracy=%.4f, AUC=%.4f",
                 test_results["accuracy"], test_results["auc"])

        # Team-only evaluation: equalize rating features to 1500/1500
        # Rating features are the last 6 columns of the feature matrix
        X_test_teamonly = X_test.copy()
        n_feat = X_test_teamonly.shape[1]
        X_test_teamonly[:, n_feat - 6] = 0.75      # r1/2000 = 1500/2000
        X_test_teamonly[:, n_feat - 5] = 0.75      # r2/2000 = 1500/2000
        X_test_teamonly[:, n_feat - 4] = 0.0       # diff = 0
        X_test_teamonly[:, n_feat - 3] = 1.0       # has_both = True
        X_test_teamonly[:, n_feat - 2] = 0.75      # max/2000 = 1500/2000
        X_test_teamonly[:, n_feat - 1] = 0.0       # abs_diff = 0
        teamonly_results = xgb_model.evaluate(X_test_teamonly, y_test)
        log.info("XGBoost TEST results (TEAM-ONLY, no ratings): accuracy=%.4f, AUC=%.4f",
                 teamonly_results["accuracy"], teamonly_results["auc"])

        # Feature importance
        top_features = xgb_model.feature_importance(top_n=20)
        log.info("Top 20 feature importances: %s", top_features)

    # ---- Ensemble calibration ----
    if args.model == "both":
        log.info("=" * 60)
        log.info("CALIBRATING ENSEMBLE WEIGHTS")
        log.info("=" * 60)

        # Reconstruct the val split (same RNG as preprocessor: random.seed + random.shuffle)
        import random as _random
        _random.seed(42)
        shuffled = list(battles)
        _random.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * train_cfg.get("train_split", 0.8))
        val_end = int(n * (train_cfg.get("train_split", 0.8) + train_cfg.get("val_split", 0.1)))
        val_battles = shuffled[train_end:val_end]
        val_labels = [1.0 if b["winner"] == 1 else 0.0 for b in val_battles]

        ensemble = EnsemblePredictor(
            neural_model=model,
            xgb_model=xgb_model,
            feature_extractor=fe,
            device=str(trainer.device),
        )
        neural_w, xgb_w = ensemble.calibrate_weights(val_battles, val_labels)
        weights_path = Path(checkpoint_dir) / f"ensemble_{args.format}_weights.json"
        ensemble.save_weights(weights_path)
        log.info("Ensemble weights for %s: neural=%.2f, xgb=%.2f", args.format, neural_w, xgb_w)

    await db.close()
    log.info("Training complete!")


def _list_to_arrays(samples: list[dict]) -> dict[str, np.ndarray]:
    """Convert list of sample dicts to dict of stacked arrays."""
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


if __name__ == "__main__":
    asyncio.run(main())
