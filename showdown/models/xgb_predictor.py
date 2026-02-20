"""XGBoost-based win probability predictor using hand-crafted features."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
)

log = logging.getLogger("showdown.models.xgb_predictor")


class XGBPredictor:
    """Gradient-boosted tree win predictor using engineered features."""

    def __init__(
        self,
        n_estimators: int = 2000,
        max_depth: int = 4,
        learning_rate: float = 0.03,
        subsample: float = 0.7,
        colsample_bytree: float = 0.5,
        colsample_bylevel: float = 0.7,
        min_child_weight: int = 5,
        gamma: float = 0.1,
        reg_alpha: float = 0.5,
        reg_lambda: float = 2.0,
        early_stopping_rounds: int = 80,
        random_state: int = 42,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            early_stopping_rounds=early_stopping_rounds,
        )
        self._is_fitted = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train the XGBoost model.

        Returns dict of training metrics.
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50,
        )
        self._is_fitted = True

        # Compute training metrics
        train_pred = self.model.predict_proba(X_train)[:, 1]
        metrics = {
            "train_accuracy": accuracy_score(y_train, (train_pred > 0.5).astype(int)),
            "train_auc": roc_auc_score(y_train, train_pred),
            "train_logloss": log_loss(y_train, train_pred),
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict_proba(X_val)[:, 1]
            metrics["val_accuracy"] = accuracy_score(y_val, (val_pred > 0.5).astype(int))
            metrics["val_auc"] = roc_auc_score(y_val, val_pred)
            metrics["val_logloss"] = log_loss(y_val, val_pred)

        log.info("XGBoost training metrics: %s", metrics)
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict win probabilities. Returns array of P(team1 wins)."""
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Full evaluation on a test set."""
        probs = self.predict(X)
        preds = (probs > 0.5).astype(int)

        return {
            "accuracy": accuracy_score(y, preds),
            "auc": roc_auc_score(y, probs),
            "logloss": log_loss(y, probs),
            "report": classification_report(y, preds, output_dict=True),
        }

    def feature_importance(self, top_n: int = 30) -> list[tuple[int, float]]:
        """Return top N feature importances (index, importance)."""
        if not self._is_fitted:
            return []
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        return [(int(i), float(importances[i])) for i in indices]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(path))
        log.info("XGBoost model saved to %s", path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(str(path))
        self._is_fitted = True
        log.info("XGBoost model loaded from %s", path)
