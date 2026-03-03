"""Ensemble predictor combining neural network and XGBoost predictions."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .win_predictor import WinPredictor
from .xgb_predictor import XGBPredictor
from ..data.features import FeatureExtractor
from ..data.mechanics import (
    ability_is_intimidate, ability_is_regenerator, is_hazard_immune_item,
)
from ..utils.constants import TYPES, TYPE_TO_IDX, NUM_TYPES

log = logging.getLogger("showdown.models.ensemble")

# Max moves per Pokemon (padded)
_MAX_MOVES = 4


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

        # Pre-computed meta team features for fast evaluation
        self._meta_teams = None
        self._meta_xgb_feats = None  # shape (n_meta, n_team_features)
        self._meta_team_data = None  # pre-digested Pokemon data per meta team

        # Vectorized meta arrays for batch matchup computation
        self._meta_arrays = None  # flat numpy arrays for ALL meta Pokemon

    # ------------------------------------------------------------------
    # Meta feature pre-computation (call once at startup)
    # ------------------------------------------------------------------

    def precompute_meta_features(self, meta_teams: list[list[dict]]) -> None:
        """Pre-compute XGBoost features for all meta teams.

        Call once after loading meta teams. This eliminates redundant
        feature extraction during genetic algorithm evaluation (~50% speedup).
        """
        if not meta_teams or self.fe is None:
            return

        self._meta_teams = meta_teams

        if self.xgb_model is not None:
            self._meta_xgb_feats = np.stack([
                self.fe.team_to_engineered(team[:6])
                for team in meta_teams
            ])
            # Pre-digest all meta team Pokemon data for fast matchup features
            self._meta_team_data = [
                self.fe.precompute_team_data(team[:6])
                for team in meta_teams
            ]
            # Build vectorized numpy arrays for batch matchup computation
            self._meta_arrays = self._build_vectorized_arrays(self._meta_team_data)
            log.info(
                "Pre-computed XGBoost features for %d meta teams (%d features each, %d meta Pokemon vectorized)",
                len(meta_teams), self._meta_xgb_feats.shape[1],
                self._meta_arrays["n_total"],
            )

    def _build_vectorized_arrays(self, team_data_list: list[list[dict]]) -> dict:
        """Convert list-of-teams pre-digested data into flat numpy arrays.

        Enables vectorized threat matrix computation across ALL meta Pokemon.
        Includes item/ability modifier arrays for mechanically accurate damage.
        """
        all_pokemon = []
        team_starts = []
        team_size = 6
        for td in team_data_list:
            team_starts.append(len(all_pokemon))
            all_pokemon.extend(td)

        N = len(all_pokemon)
        team_starts = np.array(team_starts, dtype=np.int32)

        spe = np.array([p["spe"] for p in all_pokemon], dtype=np.float32)
        atk = np.array([p["atk"] for p in all_pokemon], dtype=np.float32)
        spa = np.array([p["spa"] for p in all_pokemon], dtype=np.float32)
        def_ = np.array([max(p["def"], 1) for p in all_pokemon], dtype=np.float32)
        spd = np.array([max(p["spd"], 1) for p in all_pokemon], dtype=np.float32)
        hp_actual = np.array([p["hp_actual"] for p in all_pokemon], dtype=np.float32)
        bst = np.array([p["bst"] for p in all_pokemon], dtype=np.float32)
        sr_dmg = np.array([p["sr_dmg"] for p in all_pokemon], dtype=np.float32)
        def_eff = np.stack([p["def_eff"] for p in all_pokemon])  # (N, 18)

        # Item/ability modifier arrays
        atk_item_mult = np.array([p.get("atk_item_mult", 1.0) for p in all_pokemon], dtype=np.float32)
        spa_item_mult = np.array([p.get("spa_item_mult", 1.0) for p in all_pokemon], dtype=np.float32)
        def_item_mult = np.array([p.get("def_item_mult", 1.0) for p in all_pokemon], dtype=np.float32)
        spd_item_mult = np.array([p.get("spd_item_mult", 1.0) for p in all_pokemon], dtype=np.float32)
        damage_mult = np.array([p.get("damage_mult", 1.0) for p in all_pokemon], dtype=np.float32)
        stab_mult = np.array([p.get("stab_mult", 1.5) for p in all_pokemon], dtype=np.float32)

        # Ability type immunity indices (-1 if none)
        ability_immune_type = np.full(N, -1, dtype=np.int32)
        for i, p in enumerate(all_pokemon):
            imm = p.get("ability_immune_type")
            if imm:
                ability_immune_type[i] = TYPE_TO_IDX.get(imm, -1)

        # Move arrays padded to _MAX_MOVES
        moves_bp = np.zeros((N, _MAX_MOVES), dtype=np.float32)
        moves_type_idx = np.full((N, _MAX_MOVES), -1, dtype=np.int32)
        moves_is_physical = np.zeros((N, _MAX_MOVES), dtype=bool)
        moves_is_stab = np.zeros((N, _MAX_MOVES), dtype=bool)
        moves_is_damaging = np.zeros((N, _MAX_MOVES), dtype=bool)

        for i, p in enumerate(all_pokemon):
            for j, md in enumerate(p["moves_data"][:_MAX_MOVES]):
                bp = md["basePower"]
                if bp > 0 and md["category"] != "Status":
                    moves_bp[i, j] = bp
                    moves_type_idx[i, j] = md["type_idx"]
                    moves_is_physical[i, j] = (md["category"] == "Physical")
                    moves_is_stab[i, j] = md["is_stab"]
                    moves_is_damaging[i, j] = True

        # Priority attackers
        prio_moves = self.fe.PRIORITY_MOVES if self.fe else set()
        has_priority_atk = np.array([
            bool(p["move_ids"] & prio_moves) and (p["atk"] >= 100 or p["spa"] >= 100)
            for p in all_pokemon
        ], dtype=bool)

        # Type indices per Pokemon (for STAB checks)
        type_indices = [p["type_indices"] for p in all_pokemon]

        # Hazard move IDs, ability flags for new matchup features
        hazard_moves = self.fe.HAZARD_MOVES if self.fe else set()
        pivot_moves = self.fe.PIVOT_MOVES if self.fe else set()
        has_hazard = np.array([bool(p["move_ids"] & hazard_moves) for p in all_pokemon], dtype=bool)
        has_pivot = np.array([bool(p["move_ids"] & pivot_moves) for p in all_pokemon], dtype=bool)
        is_intimidate = np.array([ability_is_intimidate(p.get("ability_id", "")) for p in all_pokemon], dtype=bool)
        is_regenerator = np.array([ability_is_regenerator(p.get("ability_id", "")) for p in all_pokemon], dtype=bool)
        is_boots = np.array([is_hazard_immune_item(p.get("item_id", "")) for p in all_pokemon], dtype=bool)
        is_physical_atk = atk > spa  # physical attacker if atk > spa

        return {
            "n_total": N,
            "n_teams": len(team_data_list),
            "team_size": team_size,
            "team_starts": team_starts,
            "spe": spe, "atk": atk, "spa": spa,
            "def": def_, "spd": spd,
            "hp_actual": hp_actual, "bst": bst, "sr_dmg": sr_dmg,
            "def_eff": def_eff,
            "atk_item_mult": atk_item_mult, "spa_item_mult": spa_item_mult,
            "def_item_mult": def_item_mult, "spd_item_mult": spd_item_mult,
            "damage_mult": damage_mult, "stab_mult": stab_mult,
            "ability_immune_type": ability_immune_type,
            "moves_bp": moves_bp, "moves_type_idx": moves_type_idx,
            "moves_is_physical": moves_is_physical,
            "moves_is_stab": moves_is_stab,
            "moves_is_damaging": moves_is_damaging,
            "has_priority_atk": has_priority_atk,
            "type_indices": type_indices,
            "has_hazard": has_hazard, "has_pivot": has_pivot,
            "is_intimidate": is_intimidate, "is_regenerator": is_regenerator,
            "is_boots": is_boots, "is_physical_atk": is_physical_atk,
        }

    # ------------------------------------------------------------------
    # Standard prediction (full ensemble, used for single predictions)
    # ------------------------------------------------------------------

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
        Uses batched prediction for speed.
        """
        if not opponent_teams:
            return 0.5

        battles = [
            {"team1": team, "team2": opp, "winner": 0}
            for opp in opponent_teams
        ]
        probs = self._batch_predict_ensemble(battles)
        return float(np.mean(probs))

    # ------------------------------------------------------------------
    # Fast XGBoost-only prediction (used during genetic evolution)
    # ------------------------------------------------------------------

    def predict_team_winrate_fast(
        self,
        team: list[dict],
        n_sample: int | None = None,
    ) -> float:
        """Fast XGBoost-only prediction using vectorized numpy computation.

        Uses pre-computed meta arrays and numpy broadcasting to compute
        threat matrices across ALL meta Pokemon at once, eliminating
        Python loop overhead.
        """
        if self._meta_xgb_feats is None or self.xgb_model is None or self.fe is None:
            if self._meta_teams:
                return self.predict_team_winrate(team, self._meta_teams)
            return 0.5

        n = len(self._meta_teams)

        # Compute candidate team features ONCE
        t1_feat = self.fe.team_to_engineered(team[:6])
        t1_repeated = np.tile(t1_feat, (n, 1))
        diff = t1_repeated - self._meta_xgb_feats

        # Pre-digest candidate team data
        t1_data = self.fe.precompute_team_data(team[:6])

        # Vectorized matchup features across ALL meta teams at once
        matchup_feats = self._compute_matchup_batch(t1_data, self._meta_arrays)

        # Rating features — equal 1500-rated players (team-only prediction)
        #   [0] r1_safe/2000 = 0.75
        #   [1] r2_safe/2000 = 0.75
        #   [2] (r1-r2)/400  = 0.0  (equal ratings)
        #   [3] has_both      = 1.0  (asserting both ratings are known/equal)
        #   [4] max/2000      = 0.75
        #   [5] |diff|/400    = 0.0
        rating_feat = np.zeros((n, 6), dtype=np.float32)
        rating_feat[:, 0] = 0.75
        rating_feat[:, 1] = 0.75
        rating_feat[:, 3] = 1.0
        rating_feat[:, 4] = 0.75

        features = np.concatenate(
            [t1_repeated, self._meta_xgb_feats, diff, matchup_feats, rating_feat],
            axis=1,
        )
        probs = self.xgb_model.predict(features)
        return float(np.mean(probs))

    def _compute_matchup_batch(
        self, t1_data: list[dict], m: dict,
    ) -> np.ndarray:
        """Compute matchup features for candidate team vs ALL meta teams.

        Fully vectorized with item/ability modifier-aware damage formula.
        Returns: (n_teams, 30) array of matchup features.
        """
        C = len(t1_data)   # candidate team size (6)
        N = m["n_total"]   # total meta Pokemon (300)
        T = m["n_teams"]   # meta teams (50)
        S = m["team_size"] # team size (6)

        # --- Build candidate arrays ---
        c_spe = np.array([p["spe"] for p in t1_data], dtype=np.float32)
        c_atk = np.array([p["atk"] for p in t1_data], dtype=np.float32)
        c_spa = np.array([p["spa"] for p in t1_data], dtype=np.float32)
        c_def = np.array([max(p["def"], 1) for p in t1_data], dtype=np.float32)
        c_spd = np.array([max(p["spd"], 1) for p in t1_data], dtype=np.float32)
        c_hp = np.array([p["hp_actual"] for p in t1_data], dtype=np.float32)
        c_bst = np.array([p["bst"] for p in t1_data], dtype=np.float32)
        c_sr_mean = np.mean([p["sr_dmg"] for p in t1_data]).astype(np.float32)
        c_bst_mean = c_bst.mean()
        c_def_eff = np.stack([p["def_eff"] for p in t1_data])  # (C, 18)

        # Candidate item/ability modifiers
        c_atk_mult = np.array([p.get("atk_item_mult", 1.0) for p in t1_data], dtype=np.float32)
        c_spa_mult = np.array([p.get("spa_item_mult", 1.0) for p in t1_data], dtype=np.float32)
        c_def_mult = np.array([p.get("def_item_mult", 1.0) for p in t1_data], dtype=np.float32)
        c_spd_mult = np.array([p.get("spd_item_mult", 1.0) for p in t1_data], dtype=np.float32)
        c_dmg_mult = np.array([p.get("damage_mult", 1.0) for p in t1_data], dtype=np.float32)
        c_stab_mults = np.array([p.get("stab_mult", 1.5) for p in t1_data], dtype=np.float32)

        c_bp = np.zeros((C, _MAX_MOVES), dtype=np.float32)
        c_type_idx = np.full((C, _MAX_MOVES), -1, dtype=np.int32)
        c_is_phys = np.zeros((C, _MAX_MOVES), dtype=bool)
        c_is_stab = np.zeros((C, _MAX_MOVES), dtype=bool)
        c_is_dmg = np.zeros((C, _MAX_MOVES), dtype=bool)
        for i, p in enumerate(t1_data):
            for j, md in enumerate(p["moves_data"][:_MAX_MOVES]):
                bp = md["basePower"]
                if bp > 0 and md["category"] != "Status":
                    c_bp[i, j] = bp
                    c_type_idx[i, j] = md["type_idx"]
                    c_is_phys[i, j] = (md["category"] == "Physical")
                    c_is_stab[i, j] = md["is_stab"]
                    c_is_dmg[i, j] = True

        prio_moves = self.fe.PRIORITY_MOVES if self.fe else set()
        c_prio_count = sum(
            1 for p in t1_data
            if p["move_ids"] & prio_moves
            and (p["atk"] >= 100 or p["spa"] >= 100)
        )

        # Candidate STAB type indices (flattened for vectorized checks)
        c_stab_indices = []
        for i, p in enumerate(t1_data):
            for ti in p["type_indices"]:
                c_stab_indices.append(ti)
        c_stab_indices = np.array(c_stab_indices, dtype=np.int32) if c_stab_indices else np.empty(0, dtype=np.int32)

        # Candidate flags for new matchup features
        hazard_moves = self.fe.HAZARD_MOVES if self.fe else set()
        pivot_moves = self.fe.PIVOT_MOVES if self.fe else set()
        c_has_hazard = any(p["move_ids"] & hazard_moves for p in t1_data)
        c_is_intim = np.array([ability_is_intimidate(p.get("ability_id", "")) for p in t1_data])
        c_is_regen = np.array([ability_is_regenerator(p.get("ability_id", "")) for p in t1_data])
        c_is_boots = np.array([is_hazard_immune_item(p.get("item_id", "")) for p in t1_data])
        c_has_pivot = np.array([bool(p["move_ids"] & pivot_moves) for p in t1_data])
        c_is_phys_atk = c_atk > c_spa

        # Candidate ability immune type
        c_immune_type = np.full(C, -1, dtype=np.int32)
        for i, p in enumerate(t1_data):
            imm = p.get("ability_immune_type")
            if imm:
                c_immune_type[i] = TYPE_TO_IDX.get(imm, -1)

        # === VECTORIZED THREAT MATRIX: candidate (C) attacks meta (N) ===
        # Apply item/ability modifiers to attack stats
        c_a_raw = np.where(c_is_phys, c_atk[:, None], c_spa[:, None])  # (C, 4)
        c_a_mult = np.where(c_is_phys, c_atk_mult[:, None], c_spa_mult[:, None])
        c_a = c_a_raw * c_a_mult  # (C, 4)
        c_stab_per_pkmn = np.where(c_is_stab, c_stab_mults[:, None], 1.0).astype(np.float32)
        c_valid = c_is_dmg & (c_type_idx >= 0)
        c_safe_idx = np.maximum(c_type_idx, 0)

        eff_fwd = m["def_eff"][:, c_safe_idx.ravel()].reshape(N, C, _MAX_MOVES).transpose(1, 0, 2)
        # Apply defender's item defense multipliers
        d_fwd = np.where(
            c_is_phys[:, None, :],
            m["def"][None, :, None] * m["def_item_mult"][None, :, None],
            m["spd"][None, :, None] * m["spd_item_mult"][None, :, None],
        )

        dmg_fwd = ((0.84 * c_bp[:, None, :] * c_a[:, None, :] / d_fwd + 2)
                   * c_stab_per_pkmn[:, None, :] * eff_fwd
                   * c_dmg_mult[:, None, None]
                   / m["hp_actual"][None, :, None])
        dmg_fwd = np.where(c_valid[:, None, :], dmg_fwd, 0.0)

        # Zero out moves hitting ability immunities
        for i in range(C):
            for k in range(_MAX_MOVES):
                ti = c_type_idx[i, k]
                if ti >= 0:
                    mask = m["ability_immune_type"] == ti
                    dmg_fwd[i, mask, k] = 0.0

        dmg_fwd = np.minimum(dmg_fwd, 2.0)
        threat_fwd = dmg_fwd.max(axis=2)  # (C, N)

        # === REVERSE: meta (N) attacks candidate (C) ===
        m_a_raw = np.where(m["moves_is_physical"], m["atk"][:, None], m["spa"][:, None])
        m_a_mult = np.where(m["moves_is_physical"], m["atk_item_mult"][:, None], m["spa_item_mult"][:, None])
        m_a = m_a_raw * m_a_mult
        m_stab_per_pkmn = np.where(m["moves_is_stab"], m["stab_mult"][:, None], 1.0).astype(np.float32)
        m_valid = m["moves_is_damaging"] & (m["moves_type_idx"] >= 0)
        m_safe_idx = np.maximum(m["moves_type_idx"], 0)

        eff_rev = c_def_eff[:, m_safe_idx.ravel()].reshape(C, N, _MAX_MOVES).transpose(1, 0, 2)
        d_rev = np.where(
            m["moves_is_physical"][:, None, :],
            c_def[None, :, None] * c_def_mult[None, :, None],
            c_spd[None, :, None] * c_spd_mult[None, :, None],
        )

        dmg_rev = ((0.84 * m["moves_bp"][:, None, :] * m_a[:, None, :] / d_rev + 2)
                   * m_stab_per_pkmn[:, None, :] * eff_rev
                   * m["damage_mult"][:, None, None]
                   / c_hp[None, :, None])
        dmg_rev = np.where(m_valid[:, None, :], dmg_rev, 0.0)

        # Zero out moves hitting candidate ability immunities
        for j in range(C):
            imm_idx = c_immune_type[j]
            if imm_idx >= 0:
                for k in range(_MAX_MOVES):
                    mask = m["moves_type_idx"][:, k] == imm_idx
                    dmg_rev[mask, j, k] = 0.0

        dmg_rev = np.minimum(dmg_rev, 2.0)
        threat_rev = dmg_rev.max(axis=2)  # (N, C)

        # === PRE-COMPUTE per-meta-Pokemon boolean vectors ===
        se_hit_fwd = c_valid[:, None, :] & (eff_fwd >= 2.0)
        any_se_fwd = se_hit_fwd.any(axis=(0, 2))

        se_hit_rev = m_valid[:, None, :] & (eff_rev >= 2.0)
        meta_has_se_on_c = se_hit_rev.any(axis=2)

        c_any_resists = c_def_eff.min(axis=0) < 1.0
        speed_grid = c_spe[:, None] > m["spe"][None, :]
        speed_valid = (c_spe[:, None] > 0) | (m["spe"][None, :] > 0)

        # === Per-team aggregation ===
        starts = m["team_starts"]
        result = np.zeros((T, 30), dtype=np.float32)

        for t in range(T):
            s = starts[t]
            e = s + S

            thr_f = threat_fwd[:, s:e]
            thr_r = threat_rev[s:e, :]

            # 0. Speed advantage
            sv = speed_valid[:, s:e]
            sg = speed_grid[:, s:e]
            sv_sum = sv.sum()
            result[t, 0] = sg[sv].sum() / max(sv_sum, 1)

            # 1-2. SE counts
            result[t, 1] = any_se_fwd[s:e].sum() / S
            result[t, 2] = meta_has_se_on_c[s:e, :].any(axis=0).sum() / C

            # 3-4. STAB resistance
            m_ti_t = m["type_indices"][s:e]
            resist_fwd = 0
            resist_total_fwd = 0
            for j_local in range(S):
                for stab_idx in m_ti_t[j_local]:
                    resist_total_fwd += 1
                    if c_any_resists[stab_idx]:
                        resist_fwd += 1
            result[t, 3] = resist_fwd / max(resist_total_fwd, 1)

            m_de_t = m["def_eff"][s:e]
            resist_rev = 0
            resist_total_rev = 0
            for stab_idx in c_stab_indices:
                resist_total_rev += 1
                if m_de_t[:, stab_idx].min() < 1.0:
                    resist_rev += 1
            result[t, 4] = resist_rev / max(resist_total_rev, 1)

            # 5. BST advantage
            result[t, 5] = (c_bst_mean - m["bst"][s:e].mean()) / 720.0

            # 6-11. Threat matrix stats
            thr_f_flat = thr_f.ravel()
            thr_r_flat = thr_r.ravel()
            result[t, 6] = thr_f_flat.mean()
            result[t, 7] = thr_f_flat.max()
            result[t, 8] = thr_f_flat.min()
            result[t, 9] = thr_r_flat.mean()
            result[t, 10] = thr_r_flat.max()
            result[t, 11] = thr_r_flat.min()

            # 12-13. 2HKO potential
            result[t, 12] = (thr_f_flat > 0.45).sum() / max(len(thr_f_flat), 1)
            result[t, 13] = (thr_r_flat > 0.45).sum() / max(len(thr_r_flat), 1)

            # 14-15. Safe switch-in
            result[t, 14] = (thr_f.max(axis=0) < 0.25).sum() / S
            result[t, 15] = (thr_r.max(axis=0) < 0.25).sum() / C

            # 16-17. SR damage
            result[t, 16] = c_sr_mean
            result[t, 17] = m["sr_dmg"][s:e].mean()

            # 18-19. STAB immunity
            t1_immune = 0
            for stab_idx in c_stab_indices:
                if (m_de_t[:, stab_idx] == 0).any():
                    t1_immune += 1
            result[t, 18] = t1_immune / max(C * 2, 1)

            t2_immune = 0
            for j_local in range(S):
                for stab_idx in m_ti_t[j_local]:
                    if (c_def_eff[:, stab_idx] == 0).any():
                        t2_immune += 1
            result[t, 19] = t2_immune / max(S * 2, 1)

            # 20-21. Priority kill potential
            result[t, 20] = c_prio_count / 6.0
            result[t, 21] = m["has_priority_atk"][s:e].sum() / 6.0

            # ---- 22-29: Enhanced matchup features ----

            # 22-23. Ability immunity denial
            t1_denied = 0
            m_imm = m["ability_immune_type"][s:e]
            for j_local in range(S):
                imm_idx = m_imm[j_local]
                if imm_idx >= 0:
                    for i in range(C):
                        for k in range(_MAX_MOVES):
                            if c_type_idx[i, k] == imm_idx and c_bp[i, k] > 0:
                                t1_denied += 1
                                break
            t2_denied = 0
            for i in range(C):
                imm_idx = c_immune_type[i]
                if imm_idx >= 0:
                    for j_local in range(S):
                        m_idx = s + j_local
                        for k in range(_MAX_MOVES):
                            if m["moves_type_idx"][m_idx, k] == imm_idx and m["moves_bp"][m_idx, k] > 0:
                                t2_denied += 1
                                break
            result[t, 22] = t1_denied / max(C * S, 1)
            result[t, 23] = t2_denied / max(C * S, 1)

            # 24-25. Intimidate impact
            t1_intim = c_is_intim.sum()
            t2_intim = m["is_intimidate"][s:e].sum()
            t2_phys = m["is_physical_atk"][s:e].sum()
            t1_phys = c_is_phys_atk.sum()
            result[t, 24] = t1_intim * t2_phys / max(S, 1) / 6.0
            result[t, 25] = t2_intim * t1_phys / max(C, 1) / 6.0

            # 26-27. Entry hazard vs non-Boots ratio
            t2_non_boots = S - m["is_boots"][s:e].sum()
            t1_non_boots = C - c_is_boots.sum()
            result[t, 26] = t2_non_boots / S if c_has_hazard else 0.0
            m_has_hazard = m["has_hazard"][s:e].any()
            result[t, 27] = t1_non_boots / C if m_has_hazard else 0.0

            # 28-29. Regen/pivot sustainability
            c_rp = (c_is_regen & c_has_pivot).sum()
            m_rp = (m["is_regenerator"][s:e] & m["has_pivot"][s:e]).sum()
            result[t, 28] = c_rp / 6.0
            result[t, 29] = m_rp / 6.0

        return result

    # ------------------------------------------------------------------
    # Batch prediction internals
    # ------------------------------------------------------------------

    def _batch_predict_ensemble(self, battles: list[dict]) -> np.ndarray:
        """Batch-predict ensemble probabilities for multiple battles at once."""
        n = len(battles)
        neural_probs = None
        xgb_probs = None

        # Batched neural prediction
        if self.neural_model is not None and self.fe is not None:
            neural_probs = self._neural_predict_batch(battles)

        # Batched XGBoost prediction
        if self.xgb_model is not None and self.fe is not None:
            xgb_probs = self._xgb_predict_batch(battles)

        # Ensemble
        if neural_probs is not None and xgb_probs is not None:
            return self.neural_weight * neural_probs + self.xgb_weight * xgb_probs
        elif neural_probs is not None:
            return neural_probs
        elif xgb_probs is not None:
            return xgb_probs
        else:
            return np.full(n, 0.5)

    def _neural_predict(self, battle: dict) -> float:
        """Get neural network prediction for a battle."""
        return float(self._neural_predict_batch([battle])[0])

    def _neural_predict_batch(self, battles: list[dict]) -> np.ndarray:
        """Get neural network predictions for a batch of battles."""
        self.neural_model.eval()

        # Build batched tensors
        all_tensors = [self.fe.battle_to_tensors(b) for b in battles]
        keys = [k for k in all_tensors[0].keys() if k != "label"]

        batched = {}
        for k in keys:
            stacked = np.stack([t[k] for t in all_tensors], axis=0)
            batched[k] = torch.tensor(stacked, device=self.device)

        with torch.no_grad():
            probs = self.neural_model(**batched)

        return probs.cpu().numpy().flatten()

    def _xgb_predict(self, battle: dict) -> float:
        """Get XGBoost prediction for a battle."""
        return float(self._xgb_predict_batch([battle])[0])

    def _xgb_predict_batch(self, battles: list[dict]) -> np.ndarray:
        """Get XGBoost predictions for a batch of battles."""
        feats = np.stack(
            [self.fe.battle_to_engineered(b)[0] for b in battles],
            axis=0,
        )
        return self.xgb_model.predict(feats)

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

    def save_weights(self, path: str | Path) -> None:
        """Save ensemble weights to a JSON file."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "neural_weight": self.neural_weight,
                "xgb_weight": self.xgb_weight,
            }, f)
        log.info("Saved ensemble weights to %s", path)

    def load_weights(self, path: str | Path) -> bool:
        """Load ensemble weights from a JSON file. Returns True if loaded."""
        import json
        path = Path(path)
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        self.neural_weight = data["neural_weight"]
        self.xgb_weight = data["xgb_weight"]
        log.info(
            "Loaded ensemble weights: neural=%.2f, xgb=%.2f",
            self.neural_weight, self.xgb_weight,
        )
        return True
