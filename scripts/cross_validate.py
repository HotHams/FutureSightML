#!/usr/bin/env python3
"""5-Fold Cross-Validation for win prediction models.

Reports mean AUC + 95% CI for both neural and XGBoost models.

Usage:
    python scripts/cross_validate.py --format gen9ou
    python scripts/cross_validate.py --format gen9ou --model neural
    python scripts/cross_validate.py --format gen9ou --model xgboost
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from showdown.config import load_config
from showdown.data.database import Database
from showdown.data.pokemon_data import PokemonDataLoader
from showdown.data.features import FeatureExtractor
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


def _extract_neural_samples(battles: list[dict], fe: FeatureExtractor, augment: bool) -> list[dict]:
    """Convert battles to neural samples with optional augmentation."""
    samples = []
    for battle in battles:
        sample = fe.battle_to_tensors(battle)
        samples.append(sample)
        if augment:
            rf = sample["rating_features"]
            swapped_rf = rf.copy()
            swapped_rf[0] = -swapped_rf[0]
            swapped = {
                "team1_species": sample["team2_species"],
                "team1_moves": sample["team2_moves"],
                "team1_items": sample["team2_items"],
                "team1_abilities": sample["team2_abilities"],
                "team2_species": sample["team1_species"],
                "team2_moves": sample["team1_moves"],
                "team2_items": sample["team1_items"],
                "team2_abilities": sample["team1_abilities"],
                "rating_features": swapped_rf,
                "label": 1.0 - sample["label"],
            }
            if "team1_continuous" in sample:
                swapped["team1_continuous"] = sample["team2_continuous"]
                swapped["team2_continuous"] = sample["team1_continuous"]
            samples.append(swapped)
    return samples


def _extract_xgb_data(battles: list[dict], fe: FeatureExtractor, augment: bool):
    """Convert battles to XGBoost feature matrix."""
    X_list, y_list = [], []
    for battle in battles:
        feat, label = fe.battle_to_engineered(battle)
        X_list.append(feat)
        y_list.append(label)
        if augment:
            feat_s, label_s = fe.battle_to_engineered({
                "team1": battle["team2"],
                "team2": battle["team1"],
                "winner": 2 if battle["winner"] == 1 else 1,
                "rating1": battle.get("rating2"),
                "rating2": battle.get("rating1"),
            })
            X_list.append(feat_s)
            y_list.append(label_s)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


async def main():
    parser = argparse.ArgumentParser(description="5-Fold Cross-Validation")
    parser.add_argument("--format", type=str, required=True)
    parser.add_argument("--model", type=str, default="both", choices=["neural", "xgboost", "both"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    log = setup_logging("INFO")
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    xgb_cfg = cfg.get("xgboost", {})

    min_rating = args.min_rating or train_cfg.get("min_rating_filter", 1500)

    # Load data
    db_path = cfg.get("database", {}).get("path", "data/replays.db")
    db = Database(db_path)
    await db.connect()

    log.info("Loading battles for %s (min_rating=%d)...", args.format, min_rating)
    battles = await db.get_training_battles(args.format, min_rating=min_rating, limit=args.limit)
    log.info("Loaded %d battles", len(battles))

    if len(battles) < 100:
        log.error("Not enough battles (%d). Need at least 100.", len(battles))
        await db.close()
        return

    # Load Pokemon data
    pkmn_data = PokemonDataLoader()
    await pkmn_data.load()

    fe = FeatureExtractor(pokemon_data=pkmn_data)
    vocab = fe.build_vocab_from_battles(battles)
    vocab_sizes = fe.vocab_sizes

    # Get labels for stratified split (at battle level, before augmentation)
    battle_labels = np.array([1.0 if b["winner"] == 1 else 0.0 for b in battles])

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    neural_aucs = []
    neural_accs = []
    xgb_aucs = []
    xgb_accs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(battles, battle_labels)):
        log.info("=" * 60)
        log.info("FOLD %d / %d", fold_idx + 1, args.folds)
        log.info("=" * 60)

        train_battles = [battles[i] for i in train_idx]
        test_battles = [battles[i] for i in test_idx]

        # Further split train into train/val (90/10)
        n_train = len(train_battles)
        val_size = max(int(n_train * 0.1), 1)
        rng = np.random.default_rng(42 + fold_idx)
        perm = rng.permutation(n_train)
        val_battles = [train_battles[i] for i in perm[:val_size]]
        fold_train_battles = [train_battles[i] for i in perm[val_size:]]

        log.info("Split: train=%d, val=%d, test=%d",
                 len(fold_train_battles), len(val_battles), len(test_battles))

        # ---- Neural ----
        if args.model in ("neural", "both"):
            train_samples = _extract_neural_samples(fold_train_battles, fe, augment=True)
            val_samples = _extract_neural_samples(val_battles, fe, augment=False)
            test_samples = _extract_neural_samples(test_battles, fe, augment=False)

            train_arrays = _list_to_arrays(train_samples)
            val_arrays = _list_to_arrays(val_samples)
            test_arrays = _list_to_arrays(test_samples)

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

            trainer = Trainer(
                checkpoint_dir=f"data/cv_temp/fold{fold_idx}",
                device=None,
            )
            metrics = trainer.train_neural(
                model=model,
                train_data=train_arrays,
                val_data=val_arrays,
                format_id=f"{args.format}_cv{fold_idx}",
                epochs=args.epochs,
                batch_size=train_cfg.get("batch_size", 256),
                lr=train_cfg.get("learning_rate", 0.001),
                patience=train_cfg.get("patience", 10),
            )

            # Evaluate on test set
            import torch
            from torch.utils.data import DataLoader, TensorDataset

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
            fold_auc = roc_auc_score(labels_arr, preds_arr)
            fold_acc = accuracy_score(labels_arr, (preds_arr > 0.5).astype(int))
            neural_aucs.append(fold_auc)
            neural_accs.append(fold_acc)
            log.info("Fold %d Neural: AUC=%.4f, Acc=%.4f", fold_idx + 1, fold_auc, fold_acc)

        # ---- XGBoost ----
        if args.model in ("xgboost", "both"):
            X_train, y_train = _extract_xgb_data(fold_train_battles, fe, augment=True)
            X_val, y_val = _extract_xgb_data(val_battles, fe, augment=False)
            X_test, y_test = _extract_xgb_data(test_battles, fe, augment=False)

            xgb_model = XGBPredictor(
                n_estimators=xgb_cfg.get("n_estimators", 3000),
                max_depth=xgb_cfg.get("max_depth", 5),
                learning_rate=xgb_cfg.get("learning_rate", 0.02),
                subsample=xgb_cfg.get("subsample", 0.7),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.4),
                colsample_bylevel=xgb_cfg.get("colsample_bylevel", 0.7),
                min_child_weight=xgb_cfg.get("min_child_weight", 5),
                gamma=xgb_cfg.get("gamma", 0.15),
                reg_alpha=xgb_cfg.get("reg_alpha", 1.0),
                reg_lambda=xgb_cfg.get("reg_lambda", 3.0),
                early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 80),
            )

            xgb_model.train(X_train, y_train, X_val, y_val)
            test_results = xgb_model.evaluate(X_test, y_test)
            xgb_aucs.append(test_results["auc"])
            xgb_accs.append(test_results["accuracy"])
            log.info("Fold %d XGBoost: AUC=%.4f, Acc=%.4f",
                     fold_idx + 1, test_results["auc"], test_results["accuracy"])

    # ---- Summary ----
    log.info("=" * 60)
    log.info("CROSS-VALIDATION RESULTS (%d folds) - %s", args.folds, args.format)
    log.info("=" * 60)

    if neural_aucs:
        mean_auc = np.mean(neural_aucs)
        std_auc = np.std(neural_aucs)
        ci95 = 1.96 * std_auc / np.sqrt(len(neural_aucs))
        log.info("Neural AUC:  %.4f +/- %.4f  (95%% CI: [%.4f, %.4f])",
                 mean_auc, std_auc, mean_auc - ci95, mean_auc + ci95)
        log.info("Neural Acc:  %.4f +/- %.4f", np.mean(neural_accs), np.std(neural_accs))
        log.info("Per-fold AUCs: %s", [f"{a:.4f}" for a in neural_aucs])

    if xgb_aucs:
        mean_auc = np.mean(xgb_aucs)
        std_auc = np.std(xgb_aucs)
        ci95 = 1.96 * std_auc / np.sqrt(len(xgb_aucs))
        log.info("XGBoost AUC: %.4f +/- %.4f  (95%% CI: [%.4f, %.4f])",
                 mean_auc, std_auc, mean_auc - ci95, mean_auc + ci95)
        log.info("XGBoost Acc: %.4f +/- %.4f", np.mean(xgb_accs), np.std(xgb_accs))
        log.info("Per-fold AUCs: %s", [f"{a:.4f}" for a in xgb_aucs])

    # Save results
    results = {
        "format": args.format,
        "folds": args.folds,
        "n_battles": len(battles),
    }
    if neural_aucs:
        results["neural"] = {
            "mean_auc": float(np.mean(neural_aucs)),
            "std_auc": float(np.std(neural_aucs)),
            "per_fold_auc": [float(a) for a in neural_aucs],
            "mean_acc": float(np.mean(neural_accs)),
        }
    if xgb_aucs:
        results["xgboost"] = {
            "mean_auc": float(np.mean(xgb_aucs)),
            "std_auc": float(np.std(xgb_aucs)),
            "per_fold_auc": [float(a) for a in xgb_aucs],
            "mean_acc": float(np.mean(xgb_accs)),
        }

    out_path = Path("data") / f"cv_results_{args.format}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", out_path)

    # Cleanup temp checkpoints
    import shutil
    temp_dir = Path("data/cv_temp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
