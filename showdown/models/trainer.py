"""Training pipeline with checkpointing, early stopping, and metric tracking."""

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score

from .win_predictor import WinPredictor
from .xgb_predictor import XGBPredictor
from ..data.database import Database

log = logging.getLogger("showdown.models.trainer")


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by CosineAnnealingWarmRestarts.

    - Warmup phase: linearly ramp from 0 to base_lr over `warmup_epochs` epochs.
    - Cosine phase: CosineAnnealingWarmRestarts with T_0, T_mult, eta_min.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        T_0: int,
        T_mult: int = 2,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup: scale from 0 to base_lr
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing with warm restarts
            cosine_epoch = self.last_epoch - self.warmup_epochs
            # Determine which restart cycle we're in
            T_cur = self.T_0
            cycle_epoch = cosine_epoch
            while cycle_epoch >= T_cur:
                cycle_epoch -= T_cur
                T_cur *= self.T_mult
            # Cosine decay within the current cycle
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * cycle_epoch / T_cur)) / 2
                for base_lr in self.base_lrs
            ]


class Trainer:
    """Unified training pipeline for both neural and XGBoost models."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "data/checkpoints",
        device: str | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info("Trainer initialized. Device: %s", self.device)

    # ------------------------------------------------------------------
    # Neural network training
    # ------------------------------------------------------------------

    def train_neural(
        self,
        model: WinPredictor,
        train_data: dict[str, np.ndarray],
        val_data: dict[str, np.ndarray],
        format_id: str,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 0.001,
        patience: int = 10,
        db: Database | None = None,
    ) -> dict[str, Any]:
        """Train the neural win predictor.

        Args:
            model: WinPredictor model instance
            train_data: dict of numpy arrays from preprocessor
            val_data: dict of numpy arrays from preprocessor
            format_id: format string for checkpoint naming
            epochs: max training epochs
            batch_size: training batch size
            lr: learning rate
            patience: early stopping patience
            db: optional database for saving checkpoint metadata

        Returns:
            Dict of final metrics.
        """
        model = model.to(self.device)

        train_loader = self._make_dataloader(train_data, batch_size, shuffle=True)
        val_loader = self._make_dataloader(val_data, batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

        # Warmup + cosine annealing with warm restarts
        warmup_epochs = 5
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            T_0=10,
            T_mult=2,
            eta_min=lr / 50,
        )
        # Use reduction='none' so we can apply per-sample weights
        criterion = nn.BCELoss(reduction='none')

        best_val_auc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}

        log.info("Starting neural training: %d epochs, batch_size=%d, lr=%f", epochs, batch_size, lr)
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_losses = []

            for batch in train_loader:
                optimizer.zero_grad()
                inputs, labels, weights = self._unpack_batch(batch)

                # Light label smoothing (reduced since Mixup also regularizes)
                smoothed_labels = labels * 0.97 + 0.015

                # Mixup augmentation: interpolate between random pairs
                # Creates virtual training examples, reduces memorization
                lam = np.random.beta(0.2, 0.2)
                cur_batch_size = smoothed_labels.size(0)
                perm = torch.randperm(cur_batch_size, device=self.device)

                mixed_inputs = {}
                for key, val in inputs.items():
                    if val is not None and val.is_floating_point():
                        mixed_inputs[key] = lam * val + (1 - lam) * val[perm]
                    elif val is not None:
                        # For integer indices: use original (can't interpolate indices)
                        # Mixup only affects continuous features and labels
                        mixed_inputs[key] = val
                    else:
                        mixed_inputs[key] = val
                mixed_labels = lam * smoothed_labels + (1 - lam) * smoothed_labels[perm]

                # Mix weights too
                if weights is not None:
                    mixed_weights = lam * weights + (1 - lam) * weights[perm]
                else:
                    mixed_weights = None

                preds = model(**mixed_inputs)
                per_sample_loss = criterion(preds, mixed_labels)

                # Apply sample weights if available
                if mixed_weights is not None:
                    loss = (per_sample_loss * mixed_weights).mean()
                else:
                    loss = per_sample_loss.mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()

            # Validation (use hard labels for proper metric evaluation)
            val_criterion = nn.BCELoss()  # standard mean reduction for eval
            val_metrics = self._evaluate_neural(model, val_loader, val_criterion)
            train_loss = np.mean(train_losses)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_auc"].append(val_metrics["auc"])
            history["val_acc"].append(val_metrics["accuracy"])

            if epoch % 5 == 0 or epoch == 1:
                log.info(
                    "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f | val_acc=%.4f",
                    epoch, epochs, train_loss,
                    val_metrics["loss"], val_metrics["auc"], val_metrics["accuracy"],
                )

            # Always save best checkpoint
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_neural_checkpoint(model, format_id, epoch, val_metrics)
            else:
                # Only count toward patience AFTER warmup phase
                if epoch > warmup_epochs:
                    epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                log.info("Early stopping at epoch %d (best epoch: %d)", epoch, best_epoch)
                break

        elapsed = time.time() - start_time
        log.info(
            "Training complete in %.1fs. Best val AUC: %.4f at epoch %d",
            elapsed, best_val_auc, best_epoch,
        )

        # Load best checkpoint
        best_path = self.checkpoint_dir / f"neural_{format_id}_best.pt"
        if best_path.exists():
            state = torch.load(best_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state["model_state_dict"])

        final_metrics = {
            "best_val_auc": best_val_auc,
            "best_epoch": best_epoch,
            "total_epochs": epoch,
            "elapsed_seconds": elapsed,
            "history": history,
        }

        return final_metrics

    def _evaluate_neural(
        self, model: WinPredictor, loader: DataLoader, criterion: nn.Module
    ) -> dict[str, float]:
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                inputs, labels, _ = self._unpack_batch(batch)
                preds = model(**inputs)
                loss = criterion(preds, labels)
                total_loss += loss.item()
                n_batches += 1
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)

        return {
            "loss": total_loss / max(n_batches, 1),
            "auc": roc_auc_score(labels_arr, preds_arr) if len(set(labels_arr)) > 1 else 0.5,
            "accuracy": accuracy_score(labels_arr, (preds_arr > 0.5).astype(int)),
        }

    def _save_neural_checkpoint(
        self, model: WinPredictor, format_id: str, epoch: int, metrics: dict
    ) -> Path:
        path = self.checkpoint_dir / f"neural_{format_id}_best.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "format": format_id,
        }, path)
        return path

    def _make_dataloader(
        self, data: dict[str, np.ndarray], batch_size: int, shuffle: bool
    ) -> DataLoader:
        tensors = []
        keys = [
            "team1_species", "team1_moves", "team1_items", "team1_abilities",
            "team2_species", "team2_moves", "team2_items", "team2_abilities",
            "rating_features",
            "label",
        ]

        # Check if continuous features and sample weights are available
        has_continuous = "team1_continuous" in data and "team2_continuous" in data
        has_weights = "sample_weight" in data

        for key in keys:
            arr = data[key]
            if key in ("label", "rating_features"):
                tensors.append(torch.tensor(arr, dtype=torch.float32))
            else:
                tensors.append(torch.tensor(arr, dtype=torch.long))

        # Add continuous feature tensors if available
        if has_continuous:
            tensors.append(torch.tensor(data["team1_continuous"], dtype=torch.float32))
            tensors.append(torch.tensor(data["team2_continuous"], dtype=torch.float32))

        # Add sample weight tensor if available
        if has_weights:
            tensors.append(torch.tensor(data["sample_weight"], dtype=torch.float32))

        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def _unpack_batch(self, batch: tuple) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
        # Tensor count determines which optional features are present
        # Base: 10 (4 team1 + 4 team2 + rating + label)
        # +continuous: +2 (team1_continuous, team2_continuous)
        # +weights: +1 (sample_weight)
        n = len(batch)
        has_continuous = n in (12, 13)
        has_weights = n in (11, 13)

        idx = 0
        moved = [t.to(self.device) for t in batch]

        t1_species = moved[0]
        t1_moves = moved[1]
        t1_items = moved[2]
        t1_abilities = moved[3]
        t2_species = moved[4]
        t2_moves = moved[5]
        t2_items = moved[6]
        t2_abilities = moved[7]
        rating_features = moved[8]
        labels = moved[9]
        idx = 10

        if has_continuous:
            t1_continuous = moved[idx]
            t2_continuous = moved[idx + 1]
            idx += 2
        else:
            t1_continuous = None
            t2_continuous = None

        weights = moved[idx] if has_weights else None

        inputs = {
            "team1_species": t1_species,
            "team1_moves": t1_moves,
            "team1_items": t1_items,
            "team1_abilities": t1_abilities,
            "team2_species": t2_species,
            "team2_moves": t2_moves,
            "team2_items": t2_items,
            "team2_abilities": t2_abilities,
            "rating_features": rating_features,
            "team1_continuous": t1_continuous,
            "team2_continuous": t2_continuous,
        }
        return inputs, labels, weights

    # ------------------------------------------------------------------
    # XGBoost training
    # ------------------------------------------------------------------

    def train_xgboost(
        self,
        xgb_model: XGBPredictor,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        format_id: str,
        db: Database | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Train the XGBoost model and save checkpoint."""
        log.info(
            "Training XGBoost: %d train samples, %d val samples, %d features",
            len(X_train), len(X_val), X_train.shape[1],
        )

        start_time = time.time()
        metrics = xgb_model.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight)
        elapsed = time.time() - start_time

        # Save
        save_path = self.checkpoint_dir / f"xgb_{format_id}_best.joblib"
        xgb_model.save(save_path)

        metrics["elapsed_seconds"] = elapsed
        log.info("XGBoost training complete in %.1fs", elapsed)

        return metrics

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_neural(
        self, model: WinPredictor, format_id: str
    ) -> dict[str, Any] | None:
        """Load best neural checkpoint for a format."""
        path = self.checkpoint_dir / f"neural_{format_id}_best.pt"
        if not path.exists():
            log.warning("No neural checkpoint found for %s", format_id)
            return None

        state = torch.load(path, map_location=self.device, weights_only=True)
        # Validate checkpoint dimension compatibility
        saved_state = state["model_state_dict"]
        cont_key = "pokemon_encoder.continuous_proj.0.weight"
        if cont_key in saved_state:
            saved_dim = saved_state[cont_key].shape[1]
            expected_dim = model.pokemon_encoder.continuous_proj[0].in_features
            if saved_dim != expected_dim:
                log.error(
                    "Checkpoint continuous_dim mismatch for %s: "
                    "saved=%d, expected=%d. Retrain required.",
                    format_id, saved_dim, expected_dim,
                )
                return None
        model.load_state_dict(saved_state)
        model.to(self.device)
        log.info("Loaded neural checkpoint: epoch %d, metrics=%s", state["epoch"], state["metrics"])
        return state["metrics"]

    def load_xgboost(self, xgb_model: XGBPredictor, format_id: str) -> bool:
        """Load best XGBoost checkpoint for a format."""
        path = self.checkpoint_dir / f"xgb_{format_id}_best.joblib"
        if not path.exists():
            log.warning("No XGBoost checkpoint found for %s", format_id)
            return False

        xgb_model.load(path)
        return True
