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
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train the XGBoost model.

        Args:
            sample_weight: Optional per-sample weights for training data.

        Returns dict of training metrics.
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50,
            sample_weight=sample_weight,
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

    def named_feature_importance(
        self, feature_names: list[str], top_n: int = 30
    ) -> list[tuple[str, float]]:
        """Return top N feature importances with human-readable names."""
        raw = self.feature_importance(top_n=top_n)
        result = []
        for idx, imp in raw:
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            result.append((name, imp))
        return result

    def shap_importance(
        self, X: np.ndarray, feature_names: list[str] | None = None, top_n: int = 30
    ) -> dict[str, Any]:
        """Compute SHAP values for feature importance analysis.

        Args:
            X: Feature matrix (use test set for unbiased analysis).
            feature_names: Optional list of feature names.
            top_n: Number of top features to return.

        Returns:
            Dict with 'top_features' (name, mean_abs_shap) and 'shap_values' array.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        try:
            import shap
        except ImportError:
            log.warning("shap package not installed. Install with: pip install shap")
            # Fall back to native feature importance
            if feature_names:
                return {
                    "top_features": self.named_feature_importance(feature_names, top_n),
                    "method": "native_importance",
                }
            return {
                "top_features": self.feature_importance(top_n),
                "method": "native_importance",
            }

        # Use TreeExplainer for fast, exact SHAP values on tree models
        explainer = shap.TreeExplainer(self.model)

        # Subsample if dataset is large (SHAP on >10k samples is slow)
        if len(X) > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), size=5000, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        shap_values = explainer.shap_values(X_sample)

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

        top_features = []
        for idx in top_indices:
            name = (
                feature_names[idx]
                if feature_names and idx < len(feature_names)
                else f"feature_{idx}"
            )
            top_features.append((name, float(mean_abs_shap[idx])))

        return {
            "top_features": top_features,
            "shap_values": shap_values,
            "mean_abs_shap": mean_abs_shap,
            "method": "shap_tree",
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(path))
        log.info("XGBoost model saved to %s", path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(str(path))
        self._is_fitted = True
        log.info("XGBoost model loaded from %s", path)
