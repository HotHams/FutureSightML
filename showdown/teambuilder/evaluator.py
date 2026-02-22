"""Team evaluation: scores candidate teams against the metagame."""

import logging
from typing import Any

import numpy as np

from ..models.ensemble import EnsemblePredictor

log = logging.getLogger("showdown.teambuilder.evaluator")


class TeamEvaluator:
    """Evaluate team quality by predicting win rates against meta teams.

    The evaluator serves as the fitness function for the genetic algorithm.
    It scores how well a candidate team would perform against the current
    metagame by running it through the ensemble predictor against a sample
    of representative opponent teams.
    """

    def __init__(
        self,
        predictor: EnsemblePredictor,
        meta_teams: list[list[dict]],
        fast_mode: bool = False,
        n_sample: int | None = None,
    ):
        self.predictor = predictor
        self.meta_teams = meta_teams
        self.fast_mode = fast_mode
        self.n_sample = n_sample
        self._cache: dict[str, float] = {}

    def evaluate(self, team: list[dict]) -> float:
        """Score a team. Returns predicted average win rate in [0, 1].

        Higher is better. A score of 0.5 means roughly even with the meta.
        A score of 0.6 means predicted to win 60% of the time.
        """
        cache_key = self._team_key(team)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.meta_teams:
            return 0.5

        if self.fast_mode:
            winrate = self.predictor.predict_team_winrate_fast(
                team, n_sample=self.n_sample,
            )
        else:
            winrate = self.predictor.predict_team_winrate(team, self.meta_teams)

        self._cache[cache_key] = winrate
        return winrate

    def evaluate_detailed(self, team: list[dict]) -> dict[str, Any]:
        """Detailed evaluation with per-matchup breakdown."""
        if not self.meta_teams:
            return {"overall_winrate": 0.5, "matchups": []}

        matchups = []
        for opp in self.meta_teams:
            battle = {"team1": team, "team2": opp, "winner": 0}
            pred = self.predictor.predict_battle(battle)
            opp_species = [p.get("species", "?") for p in opp[:6]]
            matchups.append({
                "opponent": opp_species,
                "win_prob": pred["ensemble"],
                "neural_prob": pred.get("neural"),
                "xgb_prob": pred.get("xgboost"),
            })

        overall = np.mean([m["win_prob"] for m in matchups])

        # Sort matchups by win probability (worst first)
        matchups.sort(key=lambda m: m["win_prob"])

        return {
            "overall_winrate": float(overall),
            "worst_matchups": matchups[:5],
            "best_matchups": matchups[-5:],
            "total_matchups": len(matchups),
            "matchups_above_50": sum(1 for m in matchups if m["win_prob"] > 0.5),
        }

    def clear_cache(self) -> None:
        """Clear the evaluation cache (e.g. after meta teams update)."""
        self._cache.clear()

    @staticmethod
    def _team_key(team: list[dict]) -> str:
        """Create a hashable key for a team (order-independent)."""
        parts = []
        for p in sorted(team, key=lambda x: x.get("species", "")):
            species = p.get("species", "")
            moves = sorted(p.get("moves", []))
            item = p.get("item", "")
            ability = p.get("ability", "")
            parts.append(f"{species}:{ability}:{item}:{','.join(moves)}")
        return "|".join(parts)
