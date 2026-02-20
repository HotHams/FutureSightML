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

        Returns dict with team1_*, team2_* index arrays and the label.
        """
        t1 = self.team_to_indices(battle["team1"])
        t2 = self.team_to_indices(battle["team2"])
        label = 1.0 if battle["winner"] == 1 else 0.0

        return {
            "team1_species": t1["species"],
            "team1_moves": t1["moves"],
            "team1_items": t1["items"],
            "team1_abilities": t1["abilities"],
            "team2_species": t2["species"],
            "team2_moves": t2["moves"],
            "team2_items": t2["items"],
            "team2_abilities": t2["abilities"],
            "label": label,
        }

    # ------------------------------------------------------------------
    # Engineered features (for XGBoost)
    # ------------------------------------------------------------------

    def team_to_engineered(self, team: list[dict]) -> np.ndarray:
        """Extract hand-crafted features for a single team.

        Returns a flat feature vector. Requires pokemon_data to be loaded.
        Features:
        - Per-Pokemon type indicators (18 * 6 = 108)
        - Team type coverage offensive (18)
        - Team type coverage defensive (18)
        - Base stat aggregates (6 stats * 4 agg = 24)
        - Role indicators (6)
        - Hazard/utility flags (8)
        Total: ~182 features per team
        """
        features = []

        # Per-Pokemon type encoding
        type_matrix = np.zeros((TEAM_SIZE, NUM_TYPES), dtype=np.float32)
        base_stats = np.zeros((TEAM_SIZE, 6), dtype=np.float32)

        for i, pkmn in enumerate(team[:TEAM_SIZE]):
            species_id = _to_id(pkmn.get("species", ""))

            # Types
            types = self._get_types(species_id)
            for t in types:
                if t in TYPE_TO_IDX:
                    type_matrix[i, TYPE_TO_IDX[t]] = 1.0

            # Base stats
            stats = self._get_base_stats(species_id)
            if stats:
                for j, sn in enumerate(STAT_NAMES):
                    base_stats[i, j] = stats.get(sn, 0) / 255.0  # normalize to [0,1]

        # Flatten type matrix
        features.extend(type_matrix.flatten().tolist())

        # Offensive type coverage: for each type, does the team have SE moves?
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

        # Defensive type coverage: for each attacking type, team's best resistance
        def_coverage = np.zeros(NUM_TYPES, dtype=np.float32)
        for atk_idx, atk_type in enumerate(TYPES):
            best_resist = 4.0  # worst case
            for pkmn in team[:TEAM_SIZE]:
                types = self._get_types(_to_id(pkmn.get("species", "")))
                if types:
                    eff = type_effectiveness_against(atk_type, types)
                    best_resist = min(best_resist, eff)
            def_coverage[atk_idx] = best_resist
        features.extend(def_coverage.tolist())

        # Base stat aggregates: mean, std, min, max over team
        for stat_idx in range(6):
            stat_vals = base_stats[:len(team[:TEAM_SIZE]), stat_idx]
            if len(stat_vals) > 0:
                features.extend([
                    float(np.mean(stat_vals)),
                    float(np.std(stat_vals)),
                    float(np.min(stat_vals)),
                    float(np.max(stat_vals)),
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        # Role indicators
        roles = self._classify_roles(team[:TEAM_SIZE])
        features.extend([
            roles.get("physical_attacker", 0),
            roles.get("special_attacker", 0),
            roles.get("physical_wall", 0),
            roles.get("special_wall", 0),
            roles.get("speed_control", 0),
            roles.get("support", 0),
        ])

        # Utility flags
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
        """Extract XGBoost features for a battle. Returns (features, label)."""
        t1_feat = self.team_to_engineered(battle["team1"])
        t2_feat = self.team_to_engineered(battle["team2"])

        # Concatenate both teams + difference features
        diff = t1_feat - t2_feat
        features = np.concatenate([t1_feat, t2_feat, diff])
        label = 1.0 if battle["winner"] == 1 else 0.0
        return features, label

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
