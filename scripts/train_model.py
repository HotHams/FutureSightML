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
from showdown.data.preprocessor import DataPreprocessor
from showdown.models.win_predictor import WinPredictor
from showdown.models.xgb_predictor import XGBPredictor
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
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    log = setup_logging("INFO")
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    xgb_cfg = cfg.get("xgboost", {})

    min_rating = args.min_rating or train_cfg.get("min_rating_filter", 1500)
    epochs = args.epochs or train_cfg.get("epochs", 100)
    batch_size = args.batch_size or train_cfg.get("batch_size", 256)
    lr = args.lr or train_cfg.get("learning_rate", 0.001)
    checkpoint_dir = train_cfg.get("checkpoint_dir", "data/checkpoints")

    # Load data
    db_path = cfg.get("database", {}).get("path", "data/showdown.db")
    db = Database(db_path)
    await db.connect()

    log.info("Loading battle data for %s (min_rating=%d)...", args.format, min_rating)
    battles = await db.get_training_battles(
        args.format, min_rating=min_rating, limit=args.limit
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
    fe = FeatureExtractor(pokemon_data=pkmn_data)
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
            dropout=model_cfg.get("dropout", 0.2),
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

        log.info("Neural model results: AUC=%.4f, best_epoch=%d",
                 neural_metrics["best_val_auc"], neural_metrics["best_epoch"])

        # Test set evaluation
        test_loader_data = _list_to_arrays(neural_data["test"])
        # Saved with checkpoint already

    # ---- Train XGBoost model ----
    if args.model in ("xgboost", "both"):
        log.info("=" * 60)
        log.info("TRAINING XGBOOST MODEL")
        log.info("=" * 60)

        xgb_data = preprocessor.prepare_xgboost_dataset(battles, augment=True)
        X_train, y_train = xgb_data["train"]
        X_val, y_val = xgb_data["val"]
        X_test, y_test = xgb_data["test"]

        log.info(
            "XGBoost dataset: train=%d, val=%d, test=%d, features=%d",
            len(X_train), len(X_val), len(X_test), X_train.shape[1],
        )

        xgb_model = XGBPredictor(
            n_estimators=xgb_cfg.get("n_estimators", 1000),
            max_depth=xgb_cfg.get("max_depth", 8),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 50),
        )

        xgb_metrics = trainer.train_xgboost(
            xgb_model=xgb_model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            format_id=args.format,
            db=db,
        )

        # Test set evaluation
        test_results = xgb_model.evaluate(X_test, y_test)
        log.info("XGBoost test results: accuracy=%.4f, AUC=%.4f",
                 test_results["accuracy"], test_results["auc"])

        # Feature importance
        top_features = xgb_model.feature_importance(top_n=20)
        log.info("Top 20 feature importances: %s", top_features)

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
