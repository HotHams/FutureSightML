"""Team evaluation: scores candidate teams against the metagame."""

import logging
import re
from typing import Any

import numpy as np

from ..models.ensemble import EnsemblePredictor

log = logging.getLogger("showdown.teambuilder.evaluator")


def _to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class TeamEvaluator:
    """Evaluate team quality by predicting win rates against meta teams.

    The evaluator serves as the fitness function for the genetic algorithm.
    It scores how well a candidate team would perform against the current
    metagame by running it through the ensemble predictor against a sample
    of representative opponent teams.

    In fast_mode (used by the genetic algorithm), predictions are calibrated
    by clipping extreme model outputs and blending with a meta-alignment
    prior. This prevents the XGBoost model from wildly over-predicting
    win rates for out-of-distribution team compositions.
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

        # Pre-compute species frequency across meta teams for alignment scoring.
        # Each species gets a score = (# meta teams containing it) / (total meta teams).
        self._species_freq: dict[str, float] = {}
        if meta_teams:
            n_meta = len(meta_teams)
            counts: dict[str, int] = {}
            for team in meta_teams:
                for p in team[:6]:
                    sp = _to_id(p.get("species", ""))
                    if sp:
                        counts[sp] = counts.get(sp, 0) + 1
            self._species_freq = {sp: c / n_meta for sp, c in counts.items()}
            top_5 = sorted(self._species_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            log.info(
                "Meta-alignment prior: %d species tracked, top: %s",
                len(self._species_freq),
                ", ".join(f"{s}={r:.2f}" for s, r in top_5),
            )

    def _meta_alignment(self, team: list[dict]) -> float:
        """Compute how well a team aligns with the metagame.

        Returns the average species usage frequency across team members.
        A team of top-10 OU staples might score ~0.4, while a team of
        obscure picks would score ~0.02.
        """
        if not self._species_freq:
            return 0.5  # No meta data → neutral prior
        rates = [
            self._species_freq.get(_to_id(p.get("species", "")), 0.0)
            for p in team[:6]
        ]
        return sum(rates) / len(rates) if rates else 0.0

    def evaluate(self, team: list[dict]) -> float:
        """Score a team. Returns a fitness value for the genetic algorithm.

        In fast_mode: applies calibration (clip + meta-alignment blend)
        to prevent XGBoost extrapolation artifacts on unusual compositions.

        In standard mode: returns raw predicted win rate in [0, 1].
        """
        cache_key = self._team_key(team)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.meta_teams:
            return 0.5

        if self.fast_mode:
            raw = self.predictor.predict_team_winrate_fast(
                team, n_sample=self.n_sample,
            )
            # Calibrate: tree models extrapolate wildly for out-of-distribution
            # compositions. Clip predictions to a reasonable range, then blend
            # with meta-alignment so commonly-used Pokemon get a fitness bonus.
            clipped = float(np.clip(raw, 0.35, 0.65))
            alignment = self._meta_alignment(team)
            fitness = 0.5 * clipped + 0.5 * alignment
        else:
            fitness = self.predictor.predict_team_winrate(team, self.meta_teams)

        self._cache[cache_key] = fitness
        return fitness

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

        mean_prob = float(np.mean([m["win_prob"] for m in matchups]))
        n_favorable = sum(1 for m in matchups if m["win_prob"] > 0.5)
        matchup_winrate = n_favorable / len(matchups) if matchups else 0.5

        # Sort matchups by win probability (worst first)
        matchups.sort(key=lambda m: m["win_prob"])

        return {
            "overall_winrate": matchup_winrate,
            "mean_probability": mean_prob,
            "worst_matchups": matchups[:5],
            "best_matchups": matchups[-5:],
            "total_matchups": len(matchups),
            "matchups_above_50": n_favorable,
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
