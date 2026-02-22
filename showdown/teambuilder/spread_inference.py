"""Infer optimal EV spreads and natures from base stats, items, and movesets.

Three-tier inference:
1. Smogon usage stats — use the most common spread for the species if available
2. Item-aware templates — use the Pokemon's item to select an appropriate spread
3. Role-based templates — fall back to base-stat + moveset analysis

Since replays don't contain EV/IV/nature data, this module provides the best
possible inference using all available signals.
"""

import logging
import re
from typing import Any

from ..data.pokemon_data import PokemonDataLoader
from ..data.mechanics import ITEM_EFFECTS, is_choice_item
from ..utils.constants import NATURES, STAT_NAMES

log = logging.getLogger("showdown.teambuilder.spread_inference")


def _to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---------------------------------------------------------------------------
# Competitive EV templates (expanded from 13 → 25)
# ---------------------------------------------------------------------------
SPREADS = {
    # === Offensive sweepers ===
    "physical_sweeper":         {"nature": "Jolly",   "evs": {"atk": 252, "spe": 252, "hp": 4}},
    "physical_sweeper_power":   {"nature": "Adamant", "evs": {"atk": 252, "spe": 252, "hp": 4}},
    "special_sweeper":          {"nature": "Timid",   "evs": {"spa": 252, "spe": 252, "hp": 4}},
    "special_sweeper_power":    {"nature": "Modest",  "evs": {"spa": 252, "spe": 252, "hp": 4}},
    "mixed_attacker":           {"nature": "Naive",   "evs": {"atk": 128, "spa": 128, "spe": 252}},
    "mixed_attacker_phys":      {"nature": "Hasty",   "evs": {"atk": 252, "spa": 4, "spe": 252}},

    # === Choice item spreads (no speed nature needed for Scarf) ===
    "choice_band":              {"nature": "Adamant", "evs": {"atk": 252, "spe": 252, "hp": 4}},
    "choice_specs":             {"nature": "Modest",  "evs": {"spa": 252, "spe": 252, "hp": 4}},
    "choice_scarf_phys":        {"nature": "Jolly",   "evs": {"atk": 252, "spe": 252, "hp": 4}},
    "choice_scarf_spec":        {"nature": "Timid",   "evs": {"spa": 252, "spe": 252, "hp": 4}},

    # === Bulky offensive ===
    "bulky_physical":           {"nature": "Adamant", "evs": {"hp": 252, "atk": 252, "def": 4}},
    "bulky_special":            {"nature": "Modest",  "evs": {"hp": 252, "spa": 252, "spd": 4}},
    "bulky_physical_def":       {"nature": "Adamant", "evs": {"hp": 128, "atk": 252, "def": 128}},
    "bulky_special_spd":        {"nature": "Modest",  "evs": {"hp": 128, "spa": 252, "spd": 128}},

    # === Defensive walls ===
    "physical_wall":            {"nature": "Impish",  "evs": {"hp": 252, "def": 252, "spd": 4}},
    "physical_wall_bold":       {"nature": "Bold",    "evs": {"hp": 252, "def": 252, "spa": 4}},
    "special_wall":             {"nature": "Calm",    "evs": {"hp": 252, "spd": 252, "def": 4}},
    "special_wall_careful":     {"nature": "Careful", "evs": {"hp": 252, "spd": 252, "def": 4}},
    "mixed_wall":               {"nature": "Bold",    "evs": {"hp": 252, "def": 128, "spd": 128}},

    # === Assault Vest / Eviolite defensive attackers ===
    "av_physical":              {"nature": "Adamant", "evs": {"hp": 252, "atk": 252, "spd": 4}},
    "av_special":               {"nature": "Modest",  "evs": {"hp": 252, "spa": 252, "spd": 4}},
    "eviolite_phys_wall":       {"nature": "Impish",  "evs": {"hp": 252, "def": 252, "spd": 4}},
    "eviolite_spec_wall":       {"nature": "Calm",    "evs": {"hp": 252, "spd": 252, "def": 4}},

    # === Support / utility ===
    "fast_support":             {"nature": "Jolly",   "evs": {"hp": 252, "spe": 252, "def": 4}},
    "fast_support_timid":       {"nature": "Timid",   "evs": {"hp": 252, "spe": 252, "spa": 4}},

    # === Trick Room ===
    "trick_room_phys":          {"nature": "Brave",   "evs": {"hp": 252, "atk": 252, "def": 4}},
    "trick_room_spec":          {"nature": "Quiet",   "evs": {"hp": 252, "spa": 252, "spd": 4}},
}


# ---------------------------------------------------------------------------
# Tier 1: Smogon usage-stats-based spread inference
# ---------------------------------------------------------------------------

def _parse_spread_string(spread_str: str) -> tuple[str, dict[str, int]] | None:
    """Parse a Smogon spread string like 'Adamant:252/0/4/0/0/252'.

    Returns (nature, {hp: N, atk: N, ...}) or None on failure.
    """
    stat_keys = ["hp", "atk", "def", "spa", "spd", "spe"]
    if ":" not in spread_str:
        return None
    try:
        nature, evs_part = spread_str.split(":", 1)
        ev_vals = evs_part.split("/")
        if len(ev_vals) != 6:
            return None
        evs = {}
        for i, val in enumerate(ev_vals):
            v = int(val)
            if v > 0:
                evs[stat_keys[i]] = v
        if nature not in NATURES:
            return None
        return (nature, evs)
    except (ValueError, IndexError):
        return None


def infer_spread_from_usage(
    species_id: str,
    usage_data: dict[str, dict] | None,
) -> dict[str, Any] | None:
    """Look up the most common spread for a species from Smogon usage stats.

    Args:
        species_id: Showdown species ID (lowercase, no special chars)
        usage_data: Parsed usage data dict from StatsScraper.parse_usage_data()
                    Maps species_id -> {spreads: {spread_str: pct, ...}, ...}

    Returns:
        Dict with 'nature', 'evs', 'ivs' if found, else None.
    """
    if not usage_data:
        return None

    species_data = usage_data.get(species_id)
    if not species_data:
        return None

    spreads = species_data.get("spreads", {})
    if not spreads:
        return None

    # Sort by usage percentage descending, try each until one parses
    sorted_spreads = sorted(spreads.items(), key=lambda x: x[1], reverse=True)
    for spread_str, pct in sorted_spreads:
        parsed = _parse_spread_string(spread_str)
        if parsed:
            nature, evs = parsed
            ivs = {}
            # Brave/Quiet natures on Trick Room sets typically use 0 Spe IVs
            if nature in ("Brave", "Quiet") and evs.get("spe", 0) == 0:
                ivs["spe"] = 0
            return {"nature": nature, "evs": evs, "ivs": ivs}

    return None


# ---------------------------------------------------------------------------
# Tier 2: Item-aware spread inference
# ---------------------------------------------------------------------------

def _infer_from_item(
    pokemon: dict,
    pkmn_data: PokemonDataLoader,
    base_stats: dict[str, int],
    phys_count: int,
    spec_count: int,
) -> dict[str, Any] | None:
    """Select a spread template based on the Pokemon's held item.

    Returns spread dict or None if item doesn't constrain the spread.
    """
    item_id = _to_id(pokemon.get("item", "") or "")
    if not item_id:
        return None

    item_eff = ITEM_EFFECTS.get(item_id, {})
    atk = base_stats.get("atk", 80)
    spa = base_stats.get("spa", 80)
    is_physical = phys_count >= spec_count and atk >= spa

    # Choice Band: max power nature, max Atk + Spe
    if item_id == "choiceband":
        return {**SPREADS["choice_band"], "ivs": {}}

    # Choice Specs: max power nature, max SpA + Spe
    if item_id == "choicespecs":
        return {**SPREADS["choice_specs"], "ivs": {}}

    # Choice Scarf: max speed nature, max offensive stat + Spe
    if item_id == "choicescarf":
        if is_physical:
            return {**SPREADS["choice_scarf_phys"], "ivs": {}}
        else:
            return {**SPREADS["choice_scarf_spec"], "ivs": {}}

    # Assault Vest: HP + offensive stat, some SpD
    if item_id == "assaultvest":
        if is_physical:
            return {**SPREADS["av_physical"], "ivs": {}}
        else:
            return {**SPREADS["av_special"], "ivs": {}}

    # Eviolite: defensive spread (NFE Pokemon)
    if item_id == "eviolite":
        defn = base_stats.get("def", 80)
        spd = base_stats.get("spd", 80)
        if defn >= spd:
            return {**SPREADS["eviolite_phys_wall"], "ivs": {}}
        else:
            return {**SPREADS["eviolite_spec_wall"], "ivs": {}}

    # Life Orb: max speed + max offensive stat (standard sweeper)
    if item_id == "lifeorb":
        if is_physical:
            # Life Orb usually wants to outspeed, so speed nature
            return {**SPREADS["physical_sweeper"], "ivs": {}}
        else:
            return {**SPREADS["special_sweeper"], "ivs": {}}

    # Heavy-Duty Boots: could be anything, but on offensive mons → sweeper
    # On defensive mons → wall. Use base stats to decide.
    if item_id == "heavydutyboots":
        hp = base_stats.get("hp", 80)
        defn = base_stats.get("def", 80)
        spd = base_stats.get("spd", 80)
        bulk_total = hp + defn + spd
        off_total = atk + spa
        if bulk_total > off_total + 60:
            # Defensive HDB user (e.g. Moltres, Zapdos)
            if defn > spd + 20:
                return {**SPREADS["physical_wall_bold"], "ivs": {}}
            elif spd > defn + 20:
                return {**SPREADS["special_wall"], "ivs": {}}
            else:
                return {**SPREADS["mixed_wall"], "ivs": {}}
        # Offensive HDB user: standard sweeper
        return None  # Fall through to role-based

    # Leftovers / Black Sludge: usually defensive
    if item_id in ("leftovers", "blacksludge"):
        hp = base_stats.get("hp", 80)
        defn = base_stats.get("def", 80)
        spd = base_stats.get("spd", 80)
        spe = base_stats.get("spe", 80)
        # Could still be offensive with recovery (Dragapult with Lefties)
        if spe >= 100 and (atk >= 100 or spa >= 100):
            return None  # Fall through
        if defn > spd + 20:
            return {**SPREADS["physical_wall"], "ivs": {}}
        elif spd > defn + 20:
            return {**SPREADS["special_wall"], "ivs": {}}
        else:
            return {**SPREADS["mixed_wall"], "ivs": {}}

    # Rocky Helmet: physical wall
    if item_id == "rockyhelmet":
        return {**SPREADS["physical_wall"], "ivs": {}}

    # Focus Sash: max speed sweeper (typically a lead)
    if item_id == "focussash":
        if is_physical:
            return {**SPREADS["physical_sweeper"], "ivs": {}}
        else:
            return {**SPREADS["special_sweeper"], "ivs": {}}

    # Booster Energy: max speed + offensive stat
    if item_id == "boosterenergy":
        if is_physical:
            return {**SPREADS["physical_sweeper"], "ivs": {}}
        else:
            return {**SPREADS["special_sweeper"], "ivs": {}}

    return None


# ---------------------------------------------------------------------------
# Tier 3: Role-based spread inference (base stats + moves)
# ---------------------------------------------------------------------------

def _is_physical_move(move_data: dict) -> bool:
    return move_data.get("category") == "Physical"


def _is_special_move(move_data: dict) -> bool:
    return move_data.get("category") == "Special"


def _detect_move_roles(
    moves: list[str],
    pkmn_data: PokemonDataLoader,
) -> dict[str, Any]:
    """Analyze a Pokemon's moveset to detect its competitive role.

    Uses data-driven move sets from pkmn_data.move_sets when available,
    falls back to hardcoded sets when data hasn't been loaded.
    """
    phys_count = 0
    spec_count = 0
    has_setup = False
    has_recovery = False
    has_hazards = False
    has_pivot = False
    has_trick_room = False
    has_status = False

    # Try data-driven move sets first
    ms = getattr(pkmn_data, "move_sets", {})
    setup_set = ms.get("setup", set())
    recovery_set = ms.get("recovery", set())
    hazard_set = ms.get("hazard", set())
    pivot_set = ms.get("pivot", set())

    # Hardcoded fallbacks if data not loaded
    if not setup_set:
        setup_set = {
            "swordsdance", "dragondance", "nastyplot", "calmmind", "bulkup",
            "irondefense", "amnesia", "agility", "autotomize", "shellsmash",
            "quiverdance", "coil", "shiftgear", "bellydrum", "curse",
            "tidyup", "victorydance", "clangoroussoul",
        }
    if not recovery_set:
        recovery_set = {
            "recover", "roost", "slackoff", "softboiled", "moonlight",
            "morningsun", "synthesis", "shoreup", "milkdrink", "wish",
            "strengthsap", "rest",
        }
    if not hazard_set:
        hazard_set = {"stealthrock", "spikes", "toxicspikes", "stickyweb", "ceaselessedge"}
    if not pivot_set:
        pivot_set = {"uturn", "voltswitch", "flipturm", "teleport", "partingshot", "batonpass"}

    for move_name in moves:
        if not move_name:
            continue
        mid = _to_id(move_name)
        move_data = pkmn_data.get_move(mid)
        if move_data:
            if _is_physical_move(move_data):
                phys_count += 1
            elif _is_special_move(move_data):
                spec_count += 1
            if move_data.get("category") == "Status":
                has_status = True

        if mid in setup_set:
            has_setup = True
        if mid in recovery_set:
            has_recovery = True
        if mid in hazard_set:
            has_hazards = True
        if mid in pivot_set:
            has_pivot = True
        if mid == "trickroom":
            has_trick_room = True

    return {
        "phys_count": phys_count,
        "spec_count": spec_count,
        "has_setup": has_setup,
        "has_recovery": has_recovery,
        "has_hazards": has_hazards,
        "has_pivot": has_pivot,
        "has_trick_room": has_trick_room,
        "has_status": has_status,
    }


def _infer_from_role(
    base_stats: dict[str, int],
    roles: dict[str, Any],
) -> dict[str, Any]:
    """Select a spread template based on base stats and detected move roles."""
    atk = base_stats.get("atk", 80)
    spa = base_stats.get("spa", 80)
    spe = base_stats.get("spe", 80)
    hp = base_stats.get("hp", 80)
    defn = base_stats.get("def", 80)
    spd = base_stats.get("spd", 80)

    phys_count = roles["phys_count"]
    spec_count = roles["spec_count"]
    has_setup = roles["has_setup"]
    has_recovery = roles["has_recovery"]
    has_hazards = roles["has_hazards"]
    has_pivot = roles["has_pivot"]
    has_trick_room = roles["has_trick_room"]

    is_fast = spe >= 90
    is_slow = spe <= 50
    is_bulky = (hp + defn + spd) >= 280
    is_offensive = (atk >= 100 or spa >= 100)
    is_physical = phys_count >= spec_count and atk >= spa

    # Trick Room Pokemon
    if is_slow and has_trick_room:
        if is_physical:
            return {**SPREADS["trick_room_phys"], "ivs": {"spe": 0}}
        else:
            return {**SPREADS["trick_room_spec"], "ivs": {"spe": 0}}

    # Defensive / support Pokemon (has recovery/hazards, not offensive, is bulky)
    if (has_recovery or has_hazards) and not is_offensive and is_bulky:
        if defn > spd + 20:
            # Physically defensive — use Bold if has special moves, Impish otherwise
            if spec_count > 0:
                return {**SPREADS["physical_wall_bold"], "ivs": {}}
            return {**SPREADS["physical_wall"], "ivs": {}}
        elif spd > defn + 20:
            # Specially defensive — use Calm if no physical moves, Careful otherwise
            if phys_count > 0:
                return {**SPREADS["special_wall_careful"], "ivs": {}}
            return {**SPREADS["special_wall"], "ivs": {}}
        else:
            return {**SPREADS["mixed_wall"], "ivs": {}}

    # Bulky attackers with recovery
    if has_recovery and is_offensive:
        if is_physical:
            if is_fast:
                # Fast + bulky + recovery: slightly bulk-oriented
                return {**SPREADS["bulky_physical_def"], "ivs": {}}
            return {**SPREADS["bulky_physical"], "ivs": {}}
        else:
            if is_fast:
                return {**SPREADS["bulky_special_spd"], "ivs": {}}
            return {**SPREADS["bulky_special"], "ivs": {}}

    # Pivot + bulky (Regenerator-style defensive pivots)
    if has_pivot and is_bulky and not is_offensive:
        if defn > spd + 20:
            return {**SPREADS["physical_wall"], "ivs": {}}
        elif spd > defn + 20:
            return {**SPREADS["special_wall"], "ivs": {}}
        else:
            return {**SPREADS["mixed_wall"], "ivs": {}}

    # Fast support (hazard leads, etc.)
    if has_hazards and is_fast:
        if spec_count > phys_count:
            return {**SPREADS["fast_support_timid"], "ivs": {}}
        return {**SPREADS["fast_support"], "ivs": {}}

    # Mixed attacker (has both physical and special moves, similar stats)
    if phys_count > 0 and spec_count > 0 and abs(atk - spa) < 20:
        if atk > spa:
            return {**SPREADS["mixed_attacker_phys"], "ivs": {}}
        return {**SPREADS["mixed_attacker"], "ivs": {}}

    # Physical attacker
    if is_physical:
        if is_fast or has_setup:
            return {**SPREADS["physical_sweeper"], "ivs": {}}
        else:
            return {**SPREADS["physical_sweeper_power"], "ivs": {}}

    # Special attacker
    if is_fast or has_setup:
        return {**SPREADS["special_sweeper"], "ivs": {}}
    else:
        return {**SPREADS["special_sweeper_power"], "ivs": {}}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_spread(
    pokemon: dict,
    pkmn_data: PokemonDataLoader,
    usage_data: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Infer EV spread and nature for a Pokemon using a three-tier fallback.

    Tier 1: Smogon usage stats (most common spread for this species)
    Tier 2: Item-aware templates (Choice Band → Adamant, AV → HP/SpD, etc.)
    Tier 3: Role-based inference from base stats + movesets

    Args:
        pokemon: Pokemon dict with species, moves, item, ability, etc.
        pkmn_data: Loaded PokemonDataLoader instance.
        usage_data: Optional parsed usage data from StatsScraper.parse_usage_data().

    Returns:
        Dict with 'nature', 'evs' (dict of stat->value), 'ivs' (dict, possibly empty).
    """
    species = pokemon.get("species", "")
    species_id = _to_id(species)

    # Tier 1: Check Smogon usage stats
    usage_spread = infer_spread_from_usage(species_id, usage_data)
    if usage_spread:
        return usage_spread

    # Need base stats for tiers 2 and 3
    base_stats = pkmn_data.get_base_stats(species_id)
    if not base_stats:
        return {"nature": "Jolly", "evs": {"atk": 252, "spe": 252, "hp": 4}, "ivs": {}}

    # Analyze moves (needed for both tier 2 and 3)
    moves = pokemon.get("moves", [])
    roles = _detect_move_roles(moves, pkmn_data)

    # Tier 2: Item-aware inference
    item_spread = _infer_from_item(
        pokemon, pkmn_data, base_stats,
        roles["phys_count"], roles["spec_count"],
    )
    if item_spread:
        return item_spread

    # Tier 3: Role-based inference from base stats + moves
    return _infer_from_role(base_stats, roles)


def apply_spreads(
    team: list[dict],
    pkmn_data: PokemonDataLoader,
    usage_data: dict[str, dict] | None = None,
) -> list[dict]:
    """Apply inferred EV spreads and natures to an entire team.

    Only infers spreads for Pokemon that don't already have them
    (e.g. from Smogon stats). Pokemon with existing spreads are kept as-is.

    Args:
        team: List of Pokemon dicts.
        pkmn_data: Loaded PokemonDataLoader instance.
        usage_data: Optional parsed usage data for Tier 1 inference.

    Returns:
        List of enriched Pokemon dicts with nature, evs, and ivs.
    """
    result = []
    for pkmn in team:
        enriched = dict(pkmn)
        has_nature = pkmn.get("nature")
        has_evs = pkmn.get("evs") and any(v > 0 for v in pkmn["evs"].values())

        if not has_nature or not has_evs:
            spread = infer_spread(pkmn, pkmn_data, usage_data)
            if not has_nature:
                enriched["nature"] = spread["nature"]
            if not has_evs:
                enriched["evs"] = spread["evs"]
            if spread.get("ivs"):
                enriched.setdefault("ivs", spread["ivs"])

        result.append(enriched)
    return result
