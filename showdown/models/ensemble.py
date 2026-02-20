"""Ensemble predictor combining neural network and XGBoost predictions."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .win_predictor import WinPredictor
from .xgb_predictor import XGBPredictor
from ..data.features import FeatureExtractor

log = logging.getLogger("showdown.models.ensemble")


class EnsemblePredictor:
    """Weighted ensemble of neural and XGBoost win predictors.

    Uses a learned or fixed weighting of the two model predictions.
    The ensemble typically outperforms either model alone because:
    - Neural net captures complex embedding interactions
    - XGBoost captures explicit feature engineering signals
    """

    def __init__(
        self,
        neural_model: WinPredictor | None = None,
        xgb_model: XGBPredictor | None = None,
        feature_extractor: FeatureExtractor | None = None,
        neural_weight: float = 0.85,
        device: str = "cpu",
    ):
        self.neural_model = neural_model
        self.xgb_model = xgb_model
        self.fe = feature_extractor
        self.neural_weight = neural_weight
        self.xgb_weight = 1.0 - neural_weight
        self.device = device

    def predict_battle(self, battle: dict) -> dict[str, float]:
        """Predict win probability for a single battle.

        Args:
            battle: dict with 'team1' and 'team2' keys, each a list of pokemon dicts.

        Returns:
            Dict with 'neural', 'xgboost', 'ensemble' probabilities.
        """
        results = {}

        if self.neural_model is not None and self.fe is not None:
            results["neural"] = self._neural_predict(battle)

        if self.xgb_model is not None and self.fe is not None:
            results["xgboost"] = self._xgb_predict(battle)

        # Ensemble
        if "neural" in results and "xgboost" in results:
            results["ensemble"] = (
                self.neural_weight * results["neural"]
                + self.xgb_weight * results["xgboost"]
            )
        elif "neural" in results:
            results["ensemble"] = results["neural"]
        elif "xgboost" in results:
            results["ensemble"] = results["xgboost"]
        else:
            results["ensemble"] = 0.5

        return results

    def predict_team_winrate(
        self,
        team: list[dict],
        opponent_teams: list[list[dict]],
    ) -> float:
        """Predict average win rate of a team against a distribution of opponents.

        This is the primary method used by the team builder.
        """
        if not opponent_teams:
            return 0.5

        win_probs = []
        for opp_team in opponent_teams:
            battle = {"team1": team, "team2": opp_team, "winner": 0}
            pred = self.predict_battle(battle)
            win_probs.append(pred["ensemble"])

        return float(np.mean(win_probs))

    def _neural_predict(self, battle: dict) -> float:
        """Get neural network prediction for a battle."""
        self.neural_model.eval()
        tensors = self.fe.battle_to_tensors(battle)

        with torch.no_grad():
            inputs = {
                k: torch.tensor(v, device=self.device).unsqueeze(0)
                for k, v in tensors.items()
                if k != "label"
            }
            prob = self.neural_model(**inputs).item()

        return prob

    def _xgb_predict(self, battle: dict) -> float:
        """Get XGBoost prediction for a battle."""
        feat, _ = self.fe.battle_to_engineered(battle)
        prob = self.xgb_model.predict(feat.reshape(1, -1))[0]
        return float(prob)

    def calibrate_weights(
        self,
        battles: list[dict],
        labels: list[float],
    ) -> tuple[float, float]:
        """Find optimal ensemble weights on a validation set via grid search.

        Updates internal weights and returns (neural_weight, xgb_weight).
        """
        if self.neural_model is None or self.xgb_model is None:
            return (self.neural_weight, self.xgb_weight)

        neural_preds = []
        xgb_preds = []

        for battle in battles:
            neural_preds.append(self._neural_predict(battle))
            xgb_preds.append(self._xgb_predict(battle))

        neural_preds = np.array(neural_preds)
        xgb_preds = np.array(xgb_preds)
        labels_arr = np.array(labels)

        best_w = 0.5
        best_loss = float("inf")

        for w in np.arange(0.0, 1.05, 0.05):
            ensemble = w * neural_preds + (1 - w) * xgb_preds
            # Binary cross-entropy
            eps = 1e-7
            ensemble = np.clip(ensemble, eps, 1 - eps)
            loss = -np.mean(
                labels_arr * np.log(ensemble)
                + (1 - labels_arr) * np.log(1 - ensemble)
            )
            if loss < best_loss:
                best_loss = loss
                best_w = w

        self.neural_weight = best_w
        self.xgb_weight = 1.0 - best_w
        log.info(
            "Calibrated ensemble weights: neural=%.2f, xgb=%.2f (loss=%.4f)",
            self.neural_weight, self.xgb_weight, best_loss,
        )
        return (self.neural_weight, self.xgb_weight)
