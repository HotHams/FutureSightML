"""Training pipeline with checkpointing, early stopping, and metric tracking."""

import json
import logging
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

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCELoss()

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
                inputs, labels = self._unpack_batch(batch)
                preds = model(**inputs)
                loss = criterion(preds, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()

            # Validation
            val_metrics = self._evaluate_neural(model, val_loader, criterion)
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

            # Checkpointing
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_neural_checkpoint(model, format_id, epoch, val_metrics)
            else:
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
                inputs, labels = self._unpack_batch(batch)
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
            "label",
        ]
        for key in keys:
            arr = data[key]
            if key == "label":
                tensors.append(torch.tensor(arr, dtype=torch.float32))
            else:
                tensors.append(torch.tensor(arr, dtype=torch.long))

        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def _unpack_batch(self, batch: tuple) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        (
            t1_species, t1_moves, t1_items, t1_abilities,
            t2_species, t2_moves, t2_items, t2_abilities,
            labels,
        ) = [t.to(self.device) for t in batch]

        inputs = {
            "team1_species": t1_species,
            "team1_moves": t1_moves,
            "team1_items": t1_items,
            "team1_abilities": t1_abilities,
            "team2_species": t2_species,
            "team2_moves": t2_moves,
            "team2_items": t2_items,
            "team2_abilities": t2_abilities,
        }
        return inputs, labels

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
    ) -> dict[str, Any]:
        """Train the XGBoost model and save checkpoint."""
        log.info(
            "Training XGBoost: %d train samples, %d val samples, %d features",
            len(X_train), len(X_val), X_train.shape[1],
        )

        start_time = time.time()
        metrics = xgb_model.train(X_train, y_train, X_val, y_val)
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
        model.load_state_dict(state["model_state_dict"])
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
