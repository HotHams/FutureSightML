"""Competitive damage calculator for Pokemon (Gen 1-9).

Implements the damage formula with item, ability, weather, terrain, and
special move mechanic modifiers.  Gen-aware: accepts an optional type_chart
dict so that older-gen type matchups are computed correctly.

Formula (Gen 3+):
    damage = floor(floor(floor(2*level/5 + 2) * BP * A / D) / 50 + 2)
           * targets * weather * crit * random * STAB * type_eff
           * burn * final_mods

For feature engineering, we use a simplified "expected damage %" that
averages over the random roll and assumes no crits, no burns, and
full HP (for Multiscale etc.), giving a deterministic estimate.

References:
- https://bulbapedia.bulbagarden.net/wiki/Damage
- Pokemon Showdown engine source: sim/battle-actions.ts
"""

import math
import re
from typing import Any, Dict, Tuple

from .mechanics import (
    ITEM_EFFECTS, ABILITY_EFFECTS,
    get_ability_bp_modifier, get_item_bp_modifier,
    get_ability_stab_mult, get_ability_type_immunity,
)
from ..utils.constants import (
    TYPES, TYPE_TO_IDX,
    type_effectiveness_against,
    type_effectiveness_against_gen,
    _TYPE_CHART_GEN6PLUS,
    get_move_category,
)


def _to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# Special moves with non-standard stat usage
_OVERRIDE_OFFENSIVE_STAT = {
    "bodypress": "def",       # Uses user's Defense instead of Attack
    "bodyslam": None,         # Normal
}

_OVERRIDE_OFFENSIVE_POKEMON = {
    "foulplay": "target",     # Uses target's Attack stat
}

_OVERRIDE_DEFENSIVE_STAT = {
    "psyshock": "def",        # Special move hitting physical Defense
    "psystrike": "def",
    "secretsword": "def",
}

# Special BP modifications for specific moves
_SPECIAL_BP_MOVES = {
    "knockoff": {"bp_mult_if_item": 1.5},
    "acrobatics": {"bp_mult_no_item": 2.0},
    "facade": {"bp_mult_if_status": 2.0},
    "hex": {"bp_mult_if_status": 2.0},
    "brine": {"bp_mult_half_hp": 2.0},
    "weatherball": {"bp_in_weather": 100, "bp_base": 50},
    "terrainpulse": {"bp_in_terrain": 100, "bp_base": 50},
}


def estimate_damage_pct(
    atk_data: dict,
    def_data: dict,
    move_data: dict,
    level: int = 100,
    type_chart: Dict[Tuple[str, str], float] | None = None,
    gen: int = 9,
) -> float:
    """Estimate damage as fraction of defender's HP (0 to 2.0 cap).

    Uses pre-digested Pokemon data dicts (from precompute_team_data).
    This is the core function called by the feature pipeline.

    Args:
        atk_data: Attacker's pre-digested data dict with keys:
            atk, spa, def, spd, spe, hp_actual, types, stab_types,
            item_id, ability_id, item_effects, ability_effects,
            atk_item_mult, spa_item_mult, damage_mult, stab_mult,
            bp_flag_mults, moves_data
        def_data: Defender's pre-digested data dict (same keys)
        move_data: Move dict with basePower, category, type, flags, secondary, etc.
        level: Pokemon level (default 100)
        type_chart: Optional gen-specific type chart. If None, uses Gen 6+ chart.

    Returns:
        Estimated damage as fraction of defender HP, capped at 2.0.
    """
    bp = move_data.get("basePower", 0)
    move_type = move_data.get("type", "")
    flags = move_data.get("flags", {})

    effective_category = get_move_category(move_data, gen)
    if bp <= 0 or effective_category == "Status":
        return 0.0

    is_physical = (effective_category == "Physical")

    # --- Determine attacking stat ---
    override_off = move_data.get("overrideOffensiveStat")
    override_off_pkmn = move_data.get("overrideOffensivePokemon")

    if override_off == "def":
        # Body Press: use attacker's Defense
        a_stat = atk_data.get("def", 80)
    elif override_off_pkmn == "target":
        # Foul Play: use target's Attack
        a_stat = def_data.get("atk", 80) if is_physical else def_data.get("spa", 80)
    else:
        a_stat = atk_data.get("atk", 80) if is_physical else atk_data.get("spa", 80)

    # --- Determine defending stat ---
    override_def = move_data.get("overrideDefensiveStat")

    if override_def == "def":
        # Psyshock/Psystrike: Special move targets physical Def
        d_stat = max(def_data.get("def", 80), 1)
    else:
        d_stat = max(def_data.get("def", 80), 1) if is_physical else max(def_data.get("spd", 80), 1)

    # --- Base Power modifiers ---
    bp_effective = float(bp)

    # Ability-based BP modifiers (Technician, Iron Fist, etc.)
    atk_ability_id = atk_data.get("ability_id", "")
    has_secondary = bool(move_data.get("secondary") or move_data.get("secondaries"))
    ability_bp_mult = get_ability_bp_modifier(atk_ability_id, flags, bp, has_secondary)
    bp_effective *= ability_bp_mult

    # Item-based BP modifiers (Muscle Band, Punching Glove, etc.)
    atk_item_id = atk_data.get("item_id", "")
    item_bp_mult = get_item_bp_modifier(atk_item_id, flags, is_physical)
    bp_effective *= item_bp_mult

    # Type-boosting items (Charcoal, Mystic Water, etc.)
    atk_item_eff = atk_data.get("item_effects", {})
    if atk_item_eff.get("type_boost") == move_type:
        bp_effective *= atk_item_eff.get("type_mult", 1.0)

    # Special move-specific BP modifications
    mid = _to_id(move_data.get("name", ""))
    special = _SPECIAL_BP_MOVES.get(mid)
    if special:
        # Knock Off: 1.5x if defender has item
        if special.get("bp_mult_if_item") and def_data.get("item_id"):
            bp_effective *= special["bp_mult_if_item"]
        # Acrobatics: 2x if attacker has no item
        if special.get("bp_mult_no_item") and not atk_data.get("item_id"):
            bp_effective *= special["bp_mult_no_item"]

    # --- Attack stat modifiers ---
    a_effective = float(a_stat)

    # Item-based attack multiplier (Choice Band/Specs)
    if is_physical:
        a_effective *= atk_data.get("atk_item_mult", 1.0)
    else:
        a_effective *= atk_data.get("spa_item_mult", 1.0)

    # Ability-based attack multiplier (Huge Power, Pure Power)
    atk_ability_eff = atk_data.get("ability_effects", {})
    if is_physical and atk_ability_eff.get("atk_mult", 1.0) != 1.0:
        a_effective *= atk_ability_eff["atk_mult"]

    # --- Defense stat modifiers ---
    d_effective = float(d_stat)

    # Item-based defense multiplier (Assault Vest, Eviolite)
    if is_physical:
        d_effective *= def_data.get("def_item_mult", 1.0)
    else:
        d_effective *= def_data.get("spd_item_mult", 1.0)

    # Ability-based defense multiplier (Fur Coat)
    def_ability_eff = def_data.get("ability_effects", {})
    if is_physical and def_ability_eff.get("phys_def_mult"):
        d_effective *= def_ability_eff["phys_def_mult"]
    if not is_physical and def_ability_eff.get("special_damage_mult"):
        # Ice Scales: 0.5x special damage = effectively 2x SpD
        d_effective /= def_ability_eff["special_damage_mult"]

    # --- Core damage formula ---
    # floor(floor(floor(2*level/5 + 2) * BP * A / D) / 50 + 2)
    base_damage = math.floor(
        math.floor(
            math.floor(2 * level / 5 + 2) * bp_effective * a_effective / d_effective
        ) / 50 + 2
    )

    # --- Modifier chain ---
    modifier = 1.0

    # STAB
    stab_types = atk_data.get("stab_types", set())
    if move_type in stab_types:
        modifier *= atk_data.get("stab_mult", 1.5)

    # Type effectiveness (gen-aware)
    def_types = def_data.get("types", [])
    if def_types:
        if type_chart is not None:
            type_eff = type_effectiveness_against_gen(move_type, def_types, type_chart)
        else:
            type_eff = type_effectiveness_against(move_type, def_types)
    else:
        type_eff = 1.0

    # Check ability-based type immunity
    def_ability_id = def_data.get("ability_id", "")
    immune_type = get_ability_type_immunity(def_ability_id)
    if immune_type and move_type == immune_type:
        return 0.0  # Immune via ability (Levitate blocks Ground, etc.)

    modifier *= type_eff

    # Expert Belt: 1.2x on super effective
    if type_eff > 1.0 and atk_item_eff.get("se_damage_mult"):
        modifier *= atk_item_eff["se_damage_mult"]

    # Life Orb and other final damage multipliers
    modifier *= atk_data.get("damage_mult", 1.0)

    # Defender's super-effective resistance (Filter, Solid Rock, Prism Armor)
    if type_eff > 1.0 and def_ability_eff.get("se_damage_resist"):
        modifier *= def_ability_eff["se_damage_resist"]

    # Multiscale / Shadow Shield (assume full HP for estimation)
    if def_ability_eff.get("full_hp_damage_mult"):
        modifier *= def_ability_eff["full_hp_damage_mult"]

    # Average random roll: 0.925 (average of 0.85 to 1.0 uniform)
    modifier *= 0.925

    # Final damage
    damage = base_damage * modifier
    def_hp = max(def_data.get("hp_actual", 1), 1)
    dmg_pct = damage / def_hp

    return min(dmg_pct, 2.0)


def estimate_best_move_damage(
    atk_data: dict,
    def_data: dict,
    type_chart: Dict[Tuple[str, str], float] | None = None,
    gen: int = 9,
) -> float:
    """Estimate the best move's damage % from attacker against defender.

    Uses all moves in atk_data["moves_data"] and returns the highest.
    """
    best = 0.0
    for md in atk_data.get("moves_data", []):
        dmg = estimate_damage_pct(atk_data, def_data, md, type_chart=type_chart, gen=gen)
        if dmg > best:
            best = dmg
    return best
