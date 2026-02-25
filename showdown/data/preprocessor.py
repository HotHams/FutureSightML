"""Data preprocessing pipeline: converts raw battle records into training datasets."""

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from .features import FeatureExtractor
from .database import Database

log = logging.getLogger("showdown.data.preprocessor")


class DataPreprocessor:
    """Pipeline to convert raw battle data into train/val/test splits."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        self.fe = feature_extractor
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

    @staticmethod
    def _compute_sample_weight(battle: dict) -> float:
        """Compute sample weight based on rating quality.

        Higher-rated games get more weight since team composition
        matters more at higher Elo. Uses the higher of the two ratings.
        """
        r1 = battle.get("rating1") or 0
        r2 = battle.get("rating2") or 0
        best_rating = max(r1, r2)
        if best_rating <= 0:
            return 1.0
        return 1.0 + max(0, (best_rating - 1500)) / 500.0

    def prepare_neural_dataset(
        self, battles: list[dict], augment: bool = True
    ) -> dict[str, list[dict]]:
        """Prepare embedding-index datasets for the neural network.

        If augment=True, each battle is added twice (swapping teams + flipping label)
        to ensure the model learns symmetrically.

        IMPORTANT: Split battles FIRST, then augment each split independently
        to prevent data leakage (a battle and its mirror in different splits).

        Returns dict with 'train', 'val', 'test' keys, each a list of sample dicts.
        """
        # Split battles before augmentation to prevent leakage
        random.seed(self.seed)
        shuffled = list(battles)
        random.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))
        battle_splits = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

        result = {}
        for split_name, split_battles in battle_splits.items():
            samples = []
            for battle in split_battles:
                sample = self.fe.battle_to_tensors(battle)
                sample["sample_weight"] = self._compute_sample_weight(battle)
                samples.append(sample)

                if augment:
                    # Swap rating_features for team swap:
                    # [0] r1/2000 <-> [1] r2/2000, [2] negate diff, [3-5] symmetric
                    rf = sample["rating_features"]
                    swapped_rf = rf.copy()
                    swapped_rf[0], swapped_rf[1] = rf[1], rf[0]  # swap P1/P2 ratings
                    swapped_rf[2] = -rf[2]  # negate rating difference
                    # [3] has_both, [4] max, [5] abs_diff — unchanged (symmetric)
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
                        "sample_weight": sample["sample_weight"],
                    }
                    # Swap continuous features if present
                    if "team1_continuous" in sample:
                        swapped["team1_continuous"] = sample["team2_continuous"]
                        swapped["team2_continuous"] = sample["team1_continuous"]
                    samples.append(swapped)

            random.shuffle(samples)
            result[split_name] = samples

        return result

    def prepare_xgboost_dataset(
        self, battles: list[dict], augment: bool = True
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Prepare feature matrix + labels + sample weights for XGBoost.

        IMPORTANT: Split battles FIRST, then augment each split independently
        to prevent data leakage.

        Returns dict with 'train', 'val', 'test' keys, each a tuple of (X, y, w).
        """
        # Split battles before augmentation to prevent leakage
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(battles))
        shuffled = [battles[i] for i in indices]
        n = len(shuffled)
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))
        battle_splits = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

        result = {}
        for split_name, split_battles in battle_splits.items():
            X_list = []
            y_list = []
            w_list = []
            for battle in split_battles:
                feat, label = self.fe.battle_to_engineered(battle)
                weight = self._compute_sample_weight(battle)
                X_list.append(feat)
                y_list.append(label)
                w_list.append(weight)

                if augment:
                    feat_swapped, label_swapped = self.fe.battle_to_engineered({
                        "team1": battle["team2"],
                        "team2": battle["team1"],
                        "winner": 2 if battle["winner"] == 1 else 1,
                        "rating1": battle.get("rating2"),  # swap ratings
                        "rating2": battle.get("rating1"),
                    })
                    X_list.append(feat_swapped)
                    y_list.append(label_swapped)
                    w_list.append(weight)  # same weight for augmented copy

            X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, 0), dtype=np.float32)
            y = np.array(y_list, dtype=np.float32) if y_list else np.empty(0, dtype=np.float32)
            w = np.array(w_list, dtype=np.float32) if w_list else np.empty(0, dtype=np.float32)

            # Shuffle within split
            if len(X) > 0:
                perm = rng.permutation(len(X))
                X, y, w = X[perm], y[perm], w[perm]

            result[split_name] = (X, y, w)

        return result

    def _split(self, samples: list) -> dict[str, list]:
        n = len(samples)
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))

        return {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
            "test": samples[val_end:],
        }

    def save_processed(self, data: dict, output_dir: str | Path, prefix: str = "") -> None:
        """Save preprocessed data to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in data.items():
            if isinstance(split_data, tuple):
                # XGBoost format: (X, y)
                X, y = split_data
                np.save(output_dir / f"{prefix}{split_name}_X.npy", X)
                np.save(output_dir / f"{prefix}{split_name}_y.npy", y)
            elif isinstance(split_data, list):
                # Neural format: list of dicts with numpy arrays
                save_data = {}
                if not split_data:
                    continue
                for key in split_data[0].keys():
                    vals = [s[key] for s in split_data]
                    if isinstance(vals[0], np.ndarray):
                        save_data[key] = np.stack(vals)
                    else:
                        save_data[key] = np.array(vals)
                np.savez_compressed(
                    output_dir / f"{prefix}{split_name}.npz",
                    **save_data,
                )
                log.info(
                    "Saved %s split: %d samples -> %s",
                    split_name, len(split_data),
                    output_dir / f"{prefix}{split_name}.npz",
                )

    @staticmethod
    def load_neural_split(path: str | Path) -> dict[str, np.ndarray]:
        """Load a neural network split from .npz file."""
        data = np.load(str(path), allow_pickle=False)
        return dict(data)

    @staticmethod
    def load_xgboost_split(
        x_path: str | Path, y_path: str | Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load an XGBoost split from .npy files."""
        X = np.load(str(x_path))
        y = np.load(str(y_path))
        return X, y
