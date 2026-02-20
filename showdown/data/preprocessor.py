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

    def prepare_neural_dataset(
        self, battles: list[dict], augment: bool = True
    ) -> dict[str, list[dict]]:
        """Prepare embedding-index datasets for the neural network.

        If augment=True, each battle is added twice (swapping teams + flipping label)
        to ensure the model learns symmetrically.

        Returns dict with 'train', 'val', 'test' keys, each a list of sample dicts.
        """
        samples = []
        for battle in battles:
            sample = self.fe.battle_to_tensors(battle)
            samples.append(sample)

            if augment:
                # Swap teams and flip label
                swapped = {
                    "team1_species": sample["team2_species"],
                    "team1_moves": sample["team2_moves"],
                    "team1_items": sample["team2_items"],
                    "team1_abilities": sample["team2_abilities"],
                    "team2_species": sample["team1_species"],
                    "team2_moves": sample["team1_moves"],
                    "team2_items": sample["team1_items"],
                    "team2_abilities": sample["team1_abilities"],
                    "label": 1.0 - sample["label"],
                }
                samples.append(swapped)

        random.seed(self.seed)
        random.shuffle(samples)

        return self._split(samples)

    def prepare_xgboost_dataset(
        self, battles: list[dict], augment: bool = True
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Prepare feature matrix + labels for XGBoost.

        Returns dict with 'train', 'val', 'test' keys, each a tuple of (X, y).
        """
        X_list = []
        y_list = []

        for battle in battles:
            feat, label = self.fe.battle_to_engineered(battle)
            X_list.append(feat)
            y_list.append(label)

            if augment:
                # Swap perspective: negate difference features, flip label
                feat_swapped, label_swapped = self.fe.battle_to_engineered({
                    "team1": battle["team2"],
                    "team2": battle["team1"],
                    "winner": 2 if battle["winner"] == 1 else 1,
                })
                X_list.append(feat_swapped)
                y_list.append(label_swapped)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        # Shuffle
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(X))
        X, y = X[indices], y[indices]

        n = len(X)
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))

        return {
            "train": (X[:train_end], y[:train_end]),
            "val": (X[train_end:val_end], y[train_end:val_end]),
            "test": (X[val_end:], y[val_end:]),
        }

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
