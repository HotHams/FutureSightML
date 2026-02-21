"""Feature extraction for ML models.

Two feature modes:
1. Embedding indices — for the neural network (species/move/item/ability as integer IDs)
2. Engineered features — for XGBoost (type coverage, stat distributions, synergy scores)
"""

import logging
import re
from typing import Any

import numpy as np

from ..utils.constants import (
    TYPES, TYPE_TO_IDX, NUM_TYPES, STAT_NAMES,
    type_effectiveness_against, TEAM_SIZE,
)

log = logging.getLogger("showdown.data.features")


def _to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class FeatureExtractor:
    """Extract features from team data for both neural and XGBoost models."""

    def __init__(self, pokemon_data=None):
        """
        Args:
            pokemon_data: A loaded PokemonDataLoader instance. Required for
                         engineered features (type info, base stats).
        """
        self.pokemon_data = pokemon_data
        self._species_idx: dict[str, int] | None = None
        self._move_idx: dict[str, int] | None = None
        self._item_idx: dict[str, int] | None = None
        self._ability_idx: dict[str, int] | None = None

    def build_vocab(self) -> dict[str, dict[str, int]]:
        """Build vocabulary mappings from the loaded pokemon data.

        Returns dict with keys: species, moves, items, abilities.
        Each maps name -> integer index (0 reserved for unknown/padding).
        """
        if self.pokemon_data is None:
            raise RuntimeError("pokemon_data required to build vocab")

        self._species_idx = self.pokemon_data.species_to_idx()
        self._move_idx = self.pokemon_data.move_to_idx()
        self._item_idx = self.pokemon_data.item_to_idx()
        self._ability_idx = self.pokemon_data.ability_to_idx()

        return {
            "species": self._species_idx,
            "moves": self._move_idx,
            "items": self._item_idx,
            "abilities": self._ability_idx,
        }

    def build_vocab_from_battles(self, battles: list[dict]) -> dict[str, dict[str, int]]:
        """Build vocabulary from observed battle data (no PokemonDataLoader needed)."""
        species_set = set()
        move_set = set()
        item_set = set()
        ability_set = set()

        for battle in battles:
            for team_key in ("team1", "team2"):
                for pkmn in battle.get(team_key, []):
                    species_set.add(_to_id(pkmn.get("species", "")))
                    for m in pkmn.get("moves", []):
                        if m:
                            move_set.add(_to_id(m))
                    item = pkmn.get("item")
                    if item:
                        item_set.add(_to_id(item))
                    ability = pkmn.get("ability")
                    if ability:
                        ability_set.add(_to_id(ability))

        self._species_idx = {s: i + 1 for i, s in enumerate(sorted(species_set))}
        self._move_idx = {m: i + 1 for i, m in enumerate(sorted(move_set))}
        self._item_idx = {it: i + 1 for i, it in enumerate(sorted(item_set))}
        self._ability_idx = {a: i + 1 for i, a in enumerate(sorted(ability_set))}

        return {
            "species": self._species_idx,
            "moves": self._move_idx,
            "items": self._item_idx,
            "abilities": self._ability_idx,
        }

    @property
    def vocab_sizes(self) -> dict[str, int]:
        """Return vocab sizes (including 0-index for padding/unknown)."""
        return {
            "species": (max(self._species_idx.values()) + 1) if self._species_idx else 0,
            "moves": (max(self._move_idx.values()) + 1) if self._move_idx else 0,
            "items": (max(self._item_idx.values()) + 1) if self._item_idx else 0,
            "abilities": (max(self._ability_idx.values()) + 1) if self._ability_idx else 0,
        }

    # ------------------------------------------------------------------
    # Embedding features (for neural network)
    # ------------------------------------------------------------------

    def team_to_indices(self, team: list[dict]) -> dict[str, np.ndarray]:
        """Convert a team to embedding index arrays.

        Returns:
            {
                'species': np.array of shape (6,),
                'moves': np.array of shape (6, 4),
                'items': np.array of shape (6,),
                'abilities': np.array of shape (6,),
            }
        """
        species = np.zeros(TEAM_SIZE, dtype=np.int64)
        moves = np.zeros((TEAM_SIZE, 4), dtype=np.int64)
        items = np.zeros(TEAM_SIZE, dtype=np.int64)
        abilities = np.zeros(TEAM_SIZE, dtype=np.int64)

        for i, pkmn in enumerate(team[:TEAM_SIZE]):
            sid = _to_id(pkmn.get("species", ""))
            species[i] = self._species_idx.get(sid, 0) if self._species_idx else 0

            for j, move in enumerate(pkmn.get("moves", [])[:4]):
                mid = _to_id(move) if move else ""
                moves[i, j] = self._move_idx.get(mid, 0) if self._move_idx else 0

            item = pkmn.get("item")
            if item:
                items[i] = self._item_idx.get(_to_id(item), 0) if self._item_idx else 0

            ability = pkmn.get("ability")
            if ability:
                abilities[i] = self._ability_idx.get(_to_id(ability), 0) if self._ability_idx else 0

        return {
            "species": species,
            "moves": moves,
            "items": items,
            "abilities": abilities,
        }

    def battle_to_tensors(self, battle: dict) -> dict[str, Any]:
        """Convert a full battle record into tensor-ready data.

        Returns dict with team1_*, team2_* index arrays, rating info, and label.
        """
        t1 = self.team_to_indices(battle["team1"])
        t2 = self.team_to_indices(battle["team2"])
        label = 1.0 if battle["winner"] == 1 else 0.0

        # Rating features (normalized)
        r1 = battle.get("rating1") or 1500
        r2 = battle.get("rating2") or 1500
        rating_diff = (r1 - r2) / 400.0  # normalize: 400 ELO diff ~= 1.0
        rating_avg = ((r1 + r2) / 2 - 1500) / 400.0

        return {
            "team1_species": t1["species"],
            "team1_moves": t1["moves"],
            "team1_items": t1["items"],
            "team1_abilities": t1["abilities"],
            "team2_species": t2["species"],
            "team2_moves": t2["moves"],
            "team2_items": t2["items"],
            "team2_abilities": t2["abilities"],
            "rating_features": np.array([rating_diff, rating_avg], dtype=np.float32),
            "label": label,
        }

    # ------------------------------------------------------------------
    # Engineered features (for XGBoost)
    # ------------------------------------------------------------------

    def _team_stats_vector(self, team: list[dict]) -> dict[str, Any]:
        """Extract intermediate team data for feature engineering."""
        type_counts = np.zeros(NUM_TYPES, dtype=np.float32)
        base_stats_list = []
        speeds = []
        team_types_list = []

        for pkmn in team[:TEAM_SIZE]:
            species_id = _to_id(pkmn.get("species", ""))
            types = self._get_types(species_id)
            team_types_list.append(types)
            for t in types:
                if t in TYPE_TO_IDX:
                    type_counts[TYPE_TO_IDX[t]] += 1

            stats = self._get_base_stats(species_id)
            if stats:
                base_stats_list.append(stats)
                speeds.append(stats.get("spe", 0))
            else:
                base_stats_list.append({sn: 0 for sn in STAT_NAMES})
                speeds.append(0)

        return {
            "type_counts": type_counts,
            "base_stats_list": base_stats_list,
            "speeds": speeds,
            "types_list": team_types_list,
            "team": team[:TEAM_SIZE],
        }

    def team_to_engineered(self, team: list[dict]) -> np.ndarray:
        """Extract hand-crafted features for a single team.

        Features (team-level, compact):
        - Type composition (18): count of Pokemon per type
        - Offensive coverage (18): binary, can hit each type SE
        - Defensive coverage (18): best resistance to each attacking type
        - Base stat aggregates (6 stats * 4 agg = 24)
        - Speed tier distribution (5): count at speed brackets
        - Total BST stats (3): mean BST, min BST, max BST
        - Role indicators (6)
        - Utility flags (8)
        Total: ~100 features per team
        """
        info = self._team_stats_vector(team)
        features = []

        # Type composition (18)
        features.extend(info["type_counts"].tolist())

        # Offensive type coverage (18)
        off_coverage = np.zeros(NUM_TYPES, dtype=np.float32)
        for pkmn in team[:TEAM_SIZE]:
            for move_name in pkmn.get("moves", []):
                move_data = self._get_move(move_name)
                if move_data and move_data.get("category") != "Status":
                    move_type = move_data.get("type", "")
                    for def_idx, def_type in enumerate(TYPES):
                        eff = type_effectiveness_against(move_type, [def_type])
                        if eff >= 2.0:
                            off_coverage[def_idx] = 1.0
        features.extend(off_coverage.tolist())

        # Defensive coverage (18)
        def_coverage = np.zeros(NUM_TYPES, dtype=np.float32)
        for atk_idx, atk_type in enumerate(TYPES):
            best_resist = 4.0
            for types in info["types_list"]:
                if types:
                    eff = type_effectiveness_against(atk_type, types)
                    best_resist = min(best_resist, eff)
            def_coverage[atk_idx] = best_resist
        features.extend(def_coverage.tolist())

        # Base stat aggregates (24)
        stats_matrix = np.zeros((len(info["base_stats_list"]), 6), dtype=np.float32)
        for i, stats in enumerate(info["base_stats_list"]):
            for j, sn in enumerate(STAT_NAMES):
                stats_matrix[i, j] = stats.get(sn, 0) / 255.0
        for stat_idx in range(6):
            col = stats_matrix[:, stat_idx]
            if len(col) > 0:
                features.extend([
                    float(np.mean(col)), float(np.std(col)),
                    float(np.min(col)), float(np.max(col)),
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        # Speed tier distribution (5 brackets)
        speeds = info["speeds"]
        speed_brackets = [0, 60, 80, 100, 120]
        for i in range(len(speed_brackets)):
            lo = speed_brackets[i]
            hi = speed_brackets[i + 1] if i + 1 < len(speed_brackets) else 999
            features.append(sum(1 for s in speeds if lo <= s < hi) / max(len(speeds), 1))

        # BST aggregates (3)
        bsts = []
        for stats in info["base_stats_list"]:
            bst = sum(stats.get(sn, 0) for sn in STAT_NAMES)
            bsts.append(bst / 720.0)  # normalize by ~max BST
        if bsts:
            features.extend([float(np.mean(bsts)), float(np.min(bsts)), float(np.max(bsts))])
        else:
            features.extend([0.0, 0.0, 0.0])

        # Role indicators (6)
        roles = self._classify_roles(team[:TEAM_SIZE])
        features.extend([
            roles.get("physical_attacker", 0),
            roles.get("special_attacker", 0),
            roles.get("physical_wall", 0),
            roles.get("special_wall", 0),
            roles.get("speed_control", 0),
            roles.get("support", 0),
        ])

        # Utility flags (8)
        utility = self._utility_flags(team[:TEAM_SIZE])
        features.extend([
            utility.get("has_hazards", 0),
            utility.get("has_hazard_removal", 0),
            utility.get("has_priority", 0),
            utility.get("has_status_move", 0),
            utility.get("has_recovery", 0),
            utility.get("has_pivot", 0),
            utility.get("has_setup", 0),
            utility.get("has_choice_item", 0),
        ])

        return np.array(features, dtype=np.float32)

    def battle_to_engineered(self, battle: dict) -> tuple[np.ndarray, float]:
        """Extract XGBoost features for a battle.

        Includes per-team features + explicit matchup interaction features.
        Returns (features, label).
        """
        t1 = battle["team1"][:TEAM_SIZE]
        t2 = battle["team2"][:TEAM_SIZE]

        t1_feat = self.team_to_engineered(t1)
        t2_feat = self.team_to_engineered(t2)
        diff = t1_feat - t2_feat

        # Matchup interaction features
        matchup_feat = self._matchup_features(t1, t2)

        features = np.concatenate([t1_feat, t2_feat, diff, matchup_feat])
        label = 1.0 if battle["winner"] == 1 else 0.0
        return features, label

    def _matchup_features(self, team1: list[dict], team2: list[dict]) -> np.ndarray:
        """Compute explicit team1-vs-team2 interaction features.

        - Speed advantage ratio
        - Offensive pressure: how many of team2's Pokemon team1 can hit SE
        - Defensive resilience: how many of team1's Pokemon resist team2's STAB
        - Type advantage score
        """
        features = []

        # Speed advantages
        speeds1 = [self._get_base_stats(_to_id(p.get("species", ""))) or {} for p in team1]
        speeds2 = [self._get_base_stats(_to_id(p.get("species", ""))) or {} for p in team2]
        spe1 = [s.get("spe", 0) for s in speeds1]
        spe2 = [s.get("spe", 0) for s in speeds2]

        # Count how many of team1 outspeed how many of team2 (normalized)
        speed_wins = 0
        speed_total = 0
        for s1 in spe1:
            for s2 in spe2:
                if s1 > 0 or s2 > 0:
                    speed_total += 1
                    if s1 > s2:
                        speed_wins += 1
        features.append(speed_wins / max(speed_total, 1))

        # Offensive pressure: for each team2 Pokemon, can team1 hit it SE?
        se_count = 0
        for p2 in team2:
            types2 = self._get_types(_to_id(p2.get("species", "")))
            if not types2:
                continue
            hit_se = False
            for p1 in team1:
                for move_name in p1.get("moves", []):
                    move_data = self._get_move(move_name)
                    if move_data and move_data.get("category") != "Status":
                        eff = type_effectiveness_against(move_data.get("type", ""), types2)
                        if eff >= 2.0:
                            hit_se = True
                            break
                if hit_se:
                    break
            if hit_se:
                se_count += 1
        features.append(se_count / max(len(team2), 1))

        # Reverse: team2's offensive pressure on team1
        se_count_rev = 0
        for p1 in team1:
            types1 = self._get_types(_to_id(p1.get("species", "")))
            if not types1:
                continue
            hit_se = False
            for p2 in team2:
                for move_name in p2.get("moves", []):
                    move_data = self._get_move(move_name)
                    if move_data and move_data.get("category") != "Status":
                        eff = type_effectiveness_against(move_data.get("type", ""), types1)
                        if eff >= 2.0:
                            hit_se = True
                            break
                if hit_se:
                    break
            if hit_se:
                se_count_rev += 1
        features.append(se_count_rev / max(len(team1), 1))

        # STAB resistance: for each team2 Pokemon's STAB types, how many team1 Pokemon resist?
        resist_score = 0
        resist_total = 0
        for p2 in team2:
            types2 = self._get_types(_to_id(p2.get("species", "")))
            for stab_type in types2:
                resist_total += 1
                for p1 in team1:
                    types1 = self._get_types(_to_id(p1.get("species", "")))
                    if types1:
                        eff = type_effectiveness_against(stab_type, types1)
                        if eff < 1.0:
                            resist_score += 1
                            break
        features.append(resist_score / max(resist_total, 1))

        # Reverse: team1's STAB resisted by team2
        resist_score_rev = 0
        resist_total_rev = 0
        for p1 in team1:
            types1 = self._get_types(_to_id(p1.get("species", "")))
            for stab_type in types1:
                resist_total_rev += 1
                for p2 in team2:
                    types2 = self._get_types(_to_id(p2.get("species", "")))
                    if types2:
                        eff = type_effectiveness_against(stab_type, types2)
                        if eff < 1.0:
                            resist_score_rev += 1
                            break
        features.append(resist_score_rev / max(resist_total_rev, 1))

        # BST advantage
        bst1 = sum(sum(s.get(sn, 0) for sn in STAT_NAMES) for s in speeds1) / max(len(team1), 1)
        bst2 = sum(sum(s.get(sn, 0) for sn in STAT_NAMES) for s in speeds2) / max(len(team2), 1)
        features.append((bst1 - bst2) / 720.0)

        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_types(self, species_id: str) -> list[str]:
        if self.pokemon_data:
            return self.pokemon_data.get_types(species_id)
        return []

    def _get_base_stats(self, species_id: str) -> dict[str, int] | None:
        if self.pokemon_data:
            return self.pokemon_data.get_base_stats(species_id)
        return None

    def _get_move(self, move_name: str) -> dict | None:
        if self.pokemon_data and move_name:
            return self.pokemon_data.get_move(move_name)
        return None

    def _classify_roles(self, team: list[dict]) -> dict[str, int]:
        """Count role archetypes on the team."""
        roles = {
            "physical_attacker": 0, "special_attacker": 0,
            "physical_wall": 0, "special_wall": 0,
            "speed_control": 0, "support": 0,
        }
        for pkmn in team:
            stats = self._get_base_stats(_to_id(pkmn.get("species", "")))
            if not stats:
                continue
            atk, spa = stats.get("atk", 0), stats.get("spa", 0)
            dfn, spd = stats.get("def", 0), stats.get("spd", 0)
            spe = stats.get("spe", 0)
            hp = stats.get("hp", 0)

            if atk >= 100:
                roles["physical_attacker"] += 1
            if spa >= 100:
                roles["special_attacker"] += 1
            if dfn >= 100 and hp >= 80:
                roles["physical_wall"] += 1
            if spd >= 100 and hp >= 80:
                roles["special_wall"] += 1
            if spe >= 100:
                roles["speed_control"] += 1

            # Check for support moves
            for move_name in pkmn.get("moves", []):
                move = self._get_move(move_name)
                if move and move.get("category") == "Status":
                    roles["support"] += 1
                    break

        return roles

    HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes", "stickyweb", "ceaselessedge", "stoneaxe"}
    REMOVAL_MOVES = {"rapidspin", "defog", "courtchange", "tidyup", "mortalspinx"}
    PRIORITY_MOVES = {
        "bulletpunch", "machpunch", "aquajet", "iceshard", "shadowsneak",
        "suckerpunch", "extremespeed", "fakeout", "quickattack", "accelerock",
        "jetpunch", "grassyglide", "firstimpression",
    }
    STATUS_MOVES = {
        "thunderwave", "willowisp", "toxic", "spore", "sleeppowder",
        "stunspore", "glare", "nuzzle", "yawn",
    }
    RECOVERY_MOVES = {
        "recover", "softboiled", "roost", "moonlight", "morningsun",
        "synthesis", "shoreup", "slackoff", "wish", "rest", "strengthsap",
    }
    PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "batonpass", "shedtail"}
    SETUP_MOVES = {
        "swordsdance", "nastyplot", "dragondance", "calmmind", "bulkup",
        "irondefense", "quiverdance", "shellsmash", "agility", "autotomize",
        "bellydrum", "coil", "curse", "geomancy", "growth", "tailglow",
        "shiftgear", "tidyup", "victorydance",
    }
    CHOICE_ITEMS = {"choiceband", "choicespecs", "choicescarf"}

    def _utility_flags(self, team: list[dict]) -> dict[str, int]:
        flags = {
            "has_hazards": 0, "has_hazard_removal": 0,
            "has_priority": 0, "has_status_move": 0,
            "has_recovery": 0, "has_pivot": 0,
            "has_setup": 0, "has_choice_item": 0,
        }
        for pkmn in team:
            moves = {_to_id(m) for m in pkmn.get("moves", []) if m}
            item = _to_id(pkmn.get("item", "") or "")

            if moves & self.HAZARD_MOVES:
                flags["has_hazards"] = 1
            if moves & self.REMOVAL_MOVES:
                flags["has_hazard_removal"] = 1
            if moves & self.PRIORITY_MOVES:
                flags["has_priority"] = 1
            if moves & self.STATUS_MOVES:
                flags["has_status_move"] = 1
            if moves & self.RECOVERY_MOVES:
                flags["has_recovery"] = 1
            if moves & self.PIVOT_MOVES:
                flags["has_pivot"] = 1
            if moves & self.SETUP_MOVES:
                flags["has_setup"] = 1
            if item in self.CHOICE_ITEMS:
                flags["has_choice_item"] = 1

        return flags
