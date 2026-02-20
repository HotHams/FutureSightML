"""Format-specific constraints for legal team construction.

Enforces species clause, item clause, move legality, tier bans, etc.
"""

import logging
import re
from typing import Any

log = logging.getLogger("showdown.teambuilder.constraints")


def _to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class FormatConstraints:
    """Validates and enforces team legality rules per format."""

    # Standard Smogon clauses
    STANDARD_CLAUSES = {
        "species_clause": True,   # No duplicate species
        "item_clause": False,     # VGC: no duplicate items. Smogon: allowed
        "sleep_clause": True,
        "evasion_clause": True,
        "ohko_clause": True,
        "moody_clause": True,
        "endless_battle_clause": True,
    }

    VGC_CLAUSES = {
        "species_clause": True,
        "item_clause": True,      # VGC enforces item clause
    }

    # Known ban lists (common bans by tier — key Pokemon)
    TIER_BANS: dict[str, set[str]] = {
        "gen9ou": {
            "koraidon", "miraidon", "mewtwo", "arceus", "zacian", "zamazenta",
            "calyrexshadow", "calyrexice", "rayquaza", "kyogre", "groudon",
            "dialga", "palkia", "giratina", "reshiram", "zekrom", "kyurem",
            "xerneas", "yveltal", "zygarde", "lunala", "solgaleo",
            "necrozma", "eternatus", "terapagos", "pecharunt",
            # Common OU bans that shift over time
            "fluttermane", "palafin", "annihilape", "espathra", "ironbundle",
        },
        "gen9ubers": set(),  # Almost nothing banned in Ubers
        "gen9uu": set(),     # Populated dynamically from usage
        "gen9ru": set(),
        "gen9nu": set(),
    }

    def __init__(
        self,
        format_id: str,
        pokemon_data=None,
        banned_pokemon: set[str] | None = None,
        banned_moves: set[str] | None = None,
        banned_items: set[str] | None = None,
        banned_abilities: set[str] | None = None,
    ):
        self.format_id = format_id.lower()
        self.pokemon_data = pokemon_data
        self.is_vgc = "vgc" in self.format_id or "doubles" in self.format_id

        # Set clauses based on format
        if self.is_vgc:
            self.clauses = dict(self.VGC_CLAUSES)
        else:
            self.clauses = dict(self.STANDARD_CLAUSES)

        # Banlists
        base_bans = self.TIER_BANS.get(self.format_id, set())
        self.banned_pokemon = (banned_pokemon or set()) | base_bans
        self.banned_moves = banned_moves or set()
        self.banned_items = banned_items or set()
        self.banned_abilities = banned_abilities or set()

    def is_team_legal(self, team: list[dict]) -> tuple[bool, list[str]]:
        """Check if a team is legal. Returns (is_legal, list_of_violations)."""
        violations = []

        # Team size
        if len(team) > 6:
            violations.append(f"Team has {len(team)} Pokemon (max 6)")
        if len(team) == 0:
            violations.append("Team is empty")
            return False, violations

        # Species clause
        if self.clauses.get("species_clause"):
            species = [_to_id(p.get("species", "")) for p in team]
            # Normalize forme names to base species for species clause
            base_species = [self._base_species(s) for s in species]
            if len(set(base_species)) != len(base_species):
                violations.append("Species Clause violation: duplicate species")

        # Item clause (VGC)
        if self.clauses.get("item_clause"):
            items = [_to_id(p.get("item", "") or "") for p in team if p.get("item")]
            if len(items) != len(set(items)):
                violations.append("Item Clause violation: duplicate items")

        # Check individual Pokemon
        for i, pkmn in enumerate(team):
            species_id = _to_id(pkmn.get("species", ""))

            if species_id in self.banned_pokemon:
                violations.append(f"Banned Pokemon: {species_id}")

            ability_id = _to_id(pkmn.get("ability", "") or "")
            if ability_id in self.banned_abilities:
                violations.append(f"Banned ability: {ability_id} on {species_id}")

            item_id = _to_id(pkmn.get("item", "") or "")
            if item_id in self.banned_items:
                violations.append(f"Banned item: {item_id} on {species_id}")

            for move in pkmn.get("moves", []):
                move_id = _to_id(move) if move else ""
                if move_id in self.banned_moves:
                    violations.append(f"Banned move: {move_id} on {species_id}")

            # Check move count
            moves = [m for m in pkmn.get("moves", []) if m]
            if len(moves) > 4:
                violations.append(f"{species_id} has {len(moves)} moves (max 4)")

        return len(violations) == 0, violations

    def is_pokemon_legal(self, pkmn: dict) -> bool:
        """Quick check if a single Pokemon set is legal in this format."""
        species_id = _to_id(pkmn.get("species", ""))
        if species_id in self.banned_pokemon:
            return False

        ability_id = _to_id(pkmn.get("ability", "") or "")
        if ability_id in self.banned_abilities:
            return False

        item_id = _to_id(pkmn.get("item", "") or "")
        if item_id in self.banned_items:
            return False

        for move in pkmn.get("moves", []):
            if _to_id(move or "") in self.banned_moves:
                return False

        return True

    def filter_legal_pokemon(self, pokemon_pool: list[dict]) -> list[dict]:
        """Filter a pool of Pokemon sets to only legal ones."""
        return [p for p in pokemon_pool if self.is_pokemon_legal(p)]

    @staticmethod
    def _base_species(species_id: str) -> str:
        """Normalize formes to base species for species clause.

        e.g., 'rotomwash' -> 'rotom', 'charizardmegax' -> 'charizard'
        """
        # Common forme suffixes
        for suffix in ["mega", "megax", "megay", "gmax", "alola", "galar",
                       "hisui", "paldea", "origin", "therian", "wash",
                       "heat", "frost", "fan", "mow", "sky", "attack",
                       "defense", "speed", "sandy", "trash"]:
            if species_id.endswith(suffix) and len(species_id) > len(suffix):
                return species_id[:-len(suffix)]
        return species_id
