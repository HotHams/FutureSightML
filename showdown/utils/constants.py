"""Constants for the Pokemon Showdown ML system."""

import math
import re
from typing import Dict, Tuple

# ======================================================================
# Type system — 18 types (Gen 6+ canonical ordering)
# ======================================================================

TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]

TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}
NUM_TYPES = len(TYPES)  # always 18

# ======================================================================
# Per-generation type charts
# ======================================================================

# Gen 6+ (current) — the canonical chart already used everywhere.
_TYPE_CHART_GEN6PLUS: Dict[Tuple[str, str], float] = {
    # Normal
    ("Normal", "Rock"): 0.5, ("Normal", "Ghost"): 0.0, ("Normal", "Steel"): 0.5,
    # Fire
    ("Fire", "Fire"): 0.5, ("Fire", "Water"): 0.5, ("Fire", "Grass"): 2.0,
    ("Fire", "Ice"): 2.0, ("Fire", "Bug"): 2.0, ("Fire", "Rock"): 0.5,
    ("Fire", "Dragon"): 0.5, ("Fire", "Steel"): 2.0,
    # Water
    ("Water", "Fire"): 2.0, ("Water", "Water"): 0.5, ("Water", "Grass"): 0.5,
    ("Water", "Ground"): 2.0, ("Water", "Rock"): 2.0, ("Water", "Dragon"): 0.5,
    # Electric
    ("Electric", "Water"): 2.0, ("Electric", "Electric"): 0.5, ("Electric", "Grass"): 0.5,
    ("Electric", "Ground"): 0.0, ("Electric", "Flying"): 2.0, ("Electric", "Dragon"): 0.5,
    # Grass
    ("Grass", "Fire"): 0.5, ("Grass", "Water"): 2.0, ("Grass", "Grass"): 0.5,
    ("Grass", "Poison"): 0.5, ("Grass", "Ground"): 2.0, ("Grass", "Flying"): 0.5,
    ("Grass", "Bug"): 0.5, ("Grass", "Rock"): 2.0, ("Grass", "Dragon"): 0.5,
    ("Grass", "Steel"): 0.5,
    # Ice
    ("Ice", "Fire"): 0.5, ("Ice", "Water"): 0.5, ("Ice", "Grass"): 2.0,
    ("Ice", "Ice"): 0.5, ("Ice", "Ground"): 2.0, ("Ice", "Flying"): 2.0,
    ("Ice", "Dragon"): 2.0, ("Ice", "Steel"): 0.5,
    # Fighting
    ("Fighting", "Normal"): 2.0, ("Fighting", "Ice"): 2.0, ("Fighting", "Poison"): 0.5,
    ("Fighting", "Flying"): 0.5, ("Fighting", "Psychic"): 0.5, ("Fighting", "Bug"): 0.5,
    ("Fighting", "Rock"): 2.0, ("Fighting", "Ghost"): 0.0, ("Fighting", "Dark"): 2.0,
    ("Fighting", "Steel"): 2.0, ("Fighting", "Fairy"): 0.5,
    # Poison
    ("Poison", "Grass"): 2.0, ("Poison", "Poison"): 0.5, ("Poison", "Ground"): 0.5,
    ("Poison", "Rock"): 0.5, ("Poison", "Ghost"): 0.5, ("Poison", "Steel"): 0.0,
    ("Poison", "Fairy"): 2.0,
    # Ground
    ("Ground", "Fire"): 2.0, ("Ground", "Electric"): 2.0, ("Ground", "Grass"): 0.5,
    ("Ground", "Poison"): 2.0, ("Ground", "Flying"): 0.0, ("Ground", "Bug"): 0.5,
    ("Ground", "Rock"): 2.0, ("Ground", "Steel"): 2.0,
    # Flying
    ("Flying", "Electric"): 0.5, ("Flying", "Grass"): 2.0, ("Flying", "Fighting"): 2.0,
    ("Flying", "Bug"): 2.0, ("Flying", "Rock"): 0.5, ("Flying", "Steel"): 0.5,
    # Psychic
    ("Psychic", "Fighting"): 2.0, ("Psychic", "Poison"): 2.0, ("Psychic", "Psychic"): 0.5,
    ("Psychic", "Dark"): 0.0, ("Psychic", "Steel"): 0.5,
    # Bug
    ("Bug", "Fire"): 0.5, ("Bug", "Grass"): 2.0, ("Bug", "Fighting"): 0.5,
    ("Bug", "Poison"): 0.5, ("Bug", "Flying"): 0.5, ("Bug", "Psychic"): 2.0,
    ("Bug", "Ghost"): 0.5, ("Bug", "Dark"): 2.0, ("Bug", "Steel"): 0.5,
    ("Bug", "Fairy"): 0.5,
    # Rock
    ("Rock", "Fire"): 2.0, ("Rock", "Ice"): 2.0, ("Rock", "Fighting"): 0.5,
    ("Rock", "Ground"): 0.5, ("Rock", "Flying"): 2.0, ("Rock", "Bug"): 2.0,
    ("Rock", "Steel"): 0.5,
    # Ghost
    ("Ghost", "Normal"): 0.0, ("Ghost", "Psychic"): 2.0, ("Ghost", "Ghost"): 2.0,
    ("Ghost", "Dark"): 0.5,
    # Dragon
    ("Dragon", "Dragon"): 2.0, ("Dragon", "Steel"): 0.5, ("Dragon", "Fairy"): 0.0,
    # Dark
    ("Dark", "Fighting"): 0.5, ("Dark", "Psychic"): 2.0, ("Dark", "Ghost"): 2.0,
    ("Dark", "Dark"): 0.5, ("Dark", "Fairy"): 0.5,
    # Steel
    ("Steel", "Fire"): 0.5, ("Steel", "Water"): 0.5, ("Steel", "Electric"): 0.5,
    ("Steel", "Ice"): 2.0, ("Steel", "Rock"): 2.0, ("Steel", "Steel"): 0.5,
    ("Steel", "Fairy"): 2.0,
    # Fairy
    ("Fairy", "Fire"): 0.5, ("Fairy", "Poison"): 0.5, ("Fairy", "Fighting"): 2.0,
    ("Fairy", "Dragon"): 2.0, ("Fairy", "Dark"): 2.0, ("Fairy", "Steel"): 0.5,
}

# Gen 2-5: No Fairy type.  Steel resists Ghost and Dark.
_TYPE_CHART_GEN2TO5_OVERRIDES: Dict[Tuple[str, str], float] = {
    # Steel gained Ghost/Dark resistance in Gen 2, lost them in Gen 6
    ("Ghost", "Steel"): 0.5,
    ("Dark", "Steel"): 0.5,
    # Remove Fairy interactions (Fairy doesn't exist)
    # Any entry involving Fairy as attacker or defender becomes neutral (1.0)
    # We handle this by excluding Fairy from the valid types list for Gen 2-5
}

# Gen 1: Only 15 types (no Dark, Steel, Fairy). Several bugs/differences:
#   - Ghost has NO effect on Psychic (was a bug, intended to be SE)
#   - Poison is SE against Bug (removed in Gen 2)
#   - Bug is SE against Poison (removed in Gen 2)
_TYPE_CHART_GEN1_OVERRIDES: Dict[Tuple[str, str], float] = {
    ("Ghost", "Psychic"): 0.0,   # Gen 1 bug: Ghost immune to Psychic instead of SE
    ("Poison", "Bug"): 2.0,      # Poison SE Bug in Gen 1
    ("Bug", "Poison"): 2.0,      # Bug SE Poison in Gen 1
    # Remove these Gen 2+ entries that don't apply:
    # Ghost SE Psychic -> overridden to 0.0 above
    # Psychic 0.5x Steel -> Steel doesn't exist
    # Psychic immune Dark -> Dark doesn't exist
}

# Types available per generation era
_GEN1_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon",
]  # 15 types, no Dark/Steel/Fairy

_GEN2TO5_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel",
]  # 17 types, no Fairy


def extract_gen(format_id: str) -> int:
    """Extract generation number from a format ID string.

    Examples: 'gen9ou' -> 9, 'gen4ubers' -> 4, 'gen1ou' -> 1, 'gen31v1' -> 3
    Uses single digit match since generations are 1-9.
    """
    m = re.match(r"gen(\d)", format_id.lower())
    if m:
        return int(m.group(1))
    return 9  # default to current gen


def get_type_chart_for_gen(gen: int) -> tuple[
    list[str],                           # types list
    dict[str, int],                      # type_to_idx
    Dict[Tuple[str, str], float],        # chart overrides
    int,                                 # num_types (always 18 for feature dim)
]:
    """Return per-generation type system data.

    The types list and type_to_idx reflect which types actually exist in that
    generation, but num_types is always 18 to keep feature vectors fixed-size.
    The chart overrides dict maps (atk_type, def_type) -> multiplier.
    """
    if gen <= 1:
        valid_types = _GEN1_TYPES
        # Start with Gen 6+ chart, apply Gen 1 overrides, and strip
        # any entries involving non-existent types
        chart = {}
        for (at, dt), mult in _TYPE_CHART_GEN6PLUS.items():
            if at in valid_types and dt in valid_types:
                chart[(at, dt)] = mult
        # Apply Gen 1 specific overrides
        for (at, dt), mult in _TYPE_CHART_GEN1_OVERRIDES.items():
            if at in valid_types and dt in valid_types:
                chart[(at, dt)] = mult
    elif gen <= 5:
        valid_types = _GEN2TO5_TYPES
        chart = {}
        for (at, dt), mult in _TYPE_CHART_GEN6PLUS.items():
            if at in valid_types and dt in valid_types:
                chart[(at, dt)] = mult
        # Apply Gen 2-5 overrides (Steel resists Ghost/Dark)
        for (at, dt), mult in _TYPE_CHART_GEN2TO5_OVERRIDES.items():
            if at in valid_types and dt in valid_types:
                chart[(at, dt)] = mult
    else:
        valid_types = TYPES
        chart = _TYPE_CHART_GEN6PLUS

    valid_type_to_idx = {t: i for i, t in enumerate(valid_types)}
    return valid_types, valid_type_to_idx, chart, NUM_TYPES


def type_effectiveness_gen(atk_type: str, def_type: str,
                           chart: Dict[Tuple[str, str], float] | None = None) -> float:
    """Gen-aware type effectiveness for a single type pair."""
    if chart is None:
        chart = _TYPE_CHART_GEN6PLUS
    return chart.get((atk_type, def_type), 1.0)


def type_effectiveness_against_gen(atk_type: str, def_types: list[str],
                                   chart: Dict[Tuple[str, str], float] | None = None) -> float:
    """Gen-aware combined effectiveness against a Pokemon's type(s)."""
    if chart is None:
        chart = _TYPE_CHART_GEN6PLUS
    mult = 1.0
    for dt in def_types:
        mult *= chart.get((atk_type, dt), 1.0)
    return mult


# Keep backward-compatible module-level functions (use Gen 6+ chart)
_TYPE_CHART_OVERRIDES = _TYPE_CHART_GEN6PLUS


def type_effectiveness(atk_type: str, def_type: str) -> float:
    """Return the type effectiveness multiplier for a single attacking type vs defending type."""
    return _TYPE_CHART_OVERRIDES.get((atk_type, def_type), 1.0)


def type_effectiveness_against(atk_type: str, def_types: list[str]) -> float:
    """Return the combined effectiveness of an attacking type against a Pokemon's type(s)."""
    mult = 1.0
    for dt in def_types:
        mult *= type_effectiveness(atk_type, dt)
    return mult


NATURES = {
    "Hardy": {}, "Lonely": {"atk": 1.1, "def": 0.9},
    "Brave": {"atk": 1.1, "spe": 0.9}, "Adamant": {"atk": 1.1, "spa": 0.9},
    "Naughty": {"atk": 1.1, "spd": 0.9}, "Bold": {"def": 1.1, "atk": 0.9},
    "Docile": {}, "Relaxed": {"def": 1.1, "spe": 0.9},
    "Impish": {"def": 1.1, "spa": 0.9}, "Lax": {"def": 1.1, "spd": 0.9},
    "Timid": {"spe": 1.1, "atk": 0.9}, "Hasty": {"spe": 1.1, "def": 0.9},
    "Serious": {}, "Jolly": {"spe": 1.1, "spa": 0.9},
    "Naive": {"spe": 1.1, "spd": 0.9}, "Modest": {"spa": 1.1, "atk": 0.9},
    "Mild": {"spa": 1.1, "def": 0.9}, "Quiet": {"spa": 1.1, "spe": 0.9},
    "Bashful": {}, "Rash": {"spa": 1.1, "spd": 0.9},
    "Calm": {"spd": 1.1, "atk": 0.9}, "Gentle": {"spd": 1.1, "def": 0.9},
    "Sassy": {"spd": 1.1, "spe": 0.9}, "Careful": {"spd": 1.1, "spa": 0.9},
    "Quirky": {},
}

STAT_NAMES = ["hp", "atk", "def", "spa", "spd", "spe"]

# Move categories
PHYSICAL = "Physical"
SPECIAL = "Special"
STATUS = "Status"

# Gen 1-3: move category determined by type, not per-move.
# The classification is identical across Gen 1, 2, and 3.
# (Gen 1 simply lacks Dark and Steel; Gen 2-3 add them.)
_PRE_SPLIT_PHYSICAL_TYPES = frozenset([
    "Normal", "Fighting", "Flying", "Poison", "Ground",
    "Rock", "Bug", "Ghost", "Steel",
])
_PRE_SPLIT_SPECIAL_TYPES = frozenset([
    "Fire", "Water", "Electric", "Grass", "Ice",
    "Psychic", "Dragon", "Dark",
])


def unify_special_stat(stats: dict, gen: int) -> dict:
    """Return stats dict with Gen 1 Special stat unification applied.

    Gen 1 had a single 'Special' stat used for both offense and defense.
    The Showdown pokedex stores modern split SpA/SpD values.
    For Gen 1, we use SpA as the unified Special for both.
    Gen 2+ already have separate SpA/SpD and need no adjustment.
    """
    if gen <= 1:
        spa = stats.get("spa", stats.get("spd", 80))
        return {**stats, "spa": spa, "spd": spa}
    return stats


def get_move_category(move_data: dict, gen: int = 9) -> str:
    """Return the effective category of a move, gen-aware.

    Gen 1-3: category is determined by move type.
    Gen 4+: category is per-move (from the move data).
    """
    cat = move_data.get("category", "")
    if cat == "Status":
        return "Status"
    if gen <= 3:
        mtype = move_data.get("type", "Normal")
        if mtype in _PRE_SPLIT_PHYSICAL_TYPES:
            return "Physical"
        elif mtype in _PRE_SPLIT_SPECIAL_TYPES:
            return "Special"
    return cat

# Format types
FORMAT_SINGLES = "singles"
FORMAT_DOUBLES = "doubles"

# Map format prefixes to game type — covers all generations.
# get_game_type() fallback handles "doubles"/"vgc" in name for any we miss.
FORMAT_GAME_TYPE: dict[str, str] = {}

# Generate entries for all gens. Singles tiers:
_SINGLES_TIERS = [
    "ou", "uu", "ru", "nu", "pu", "zu", "ubers", "lc", "monotype",
    "ag", "1v1", "ubersuu", "cap",
    "battlestadiumsingles", "battlespotsingles", "battlefactorysingles",
    "customgame",
]
_DOUBLES_TIERS = [
    "doublesou", "doublesuu", "doublesru", "doublesnu", "doubleslc",
    "doublesubers", "doublesag",
    "vgc", "battlespotdoubles", "battlestadiumdoubles",
    "battlefactorydoubles",
]

for _gen in range(1, 10):
    for _tier in _SINGLES_TIERS:
        FORMAT_GAME_TYPE[f"gen{_gen}{_tier}"] = FORMAT_SINGLES
    for _tier in _DOUBLES_TIERS:
        FORMAT_GAME_TYPE[f"gen{_gen}{_tier}"] = FORMAT_DOUBLES

# Explicit entries for special/ambiguous formats
for _gen in range(1, 10):
    FORMAT_GAME_TYPE[f"gen{_gen}nationaldex"] = FORMAT_SINGLES
    FORMAT_GAME_TYPE[f"gen{_gen}nationaldexuu"] = FORMAT_SINGLES
    FORMAT_GAME_TYPE[f"gen{_gen}nationaldexubers"] = FORMAT_SINGLES
    FORMAT_GAME_TYPE[f"gen{_gen}nationaldexmonotype"] = FORMAT_SINGLES
    FORMAT_GAME_TYPE[f"gen{_gen}nationaldexag"] = FORMAT_SINGLES
    FORMAT_GAME_TYPE[f"gen{_gen}nationaldexdoubles"] = FORMAT_DOUBLES
    FORMAT_GAME_TYPE[f"gen{_gen}2v2doubles"] = FORMAT_DOUBLES

# VGC year-specific formats
for _year in range(2010, 2027):
    for _suffix in ["", "regf", "regg", "rege", "regd", "regc", "regb", "rega",
                     "series1", "series2", "series3", "series4", "series5",
                     "series6", "series7", "series8", "series9", "series10",
                     "series11", "series12", "series13"]:
        FORMAT_GAME_TYPE[f"gen9vgc{_year}{_suffix}"] = FORMAT_DOUBLES
        for _g in range(3, 9):
            FORMAT_GAME_TYPE[f"gen{_g}vgc{_year}{_suffix}"] = FORMAT_DOUBLES


def get_game_type(format_id: str) -> str:
    """Determine if a format is singles or doubles."""
    fmt_lower = format_id.lower().replace("[", "").replace("]", "").replace(" ", "")
    for prefix, game_type in FORMAT_GAME_TYPE.items():
        if fmt_lower.startswith(prefix):
            return game_type
    if "doubles" in fmt_lower or "vgc" in fmt_lower:
        return FORMAT_DOUBLES
    return FORMAT_SINGLES


# Max team size
TEAM_SIZE = 6
TEAM_SIZE_VGC_BRING = 6
TEAM_SIZE_VGC_PICK = 4

# Stat calc constants
IV_DEFAULT = 31
EV_MAX_TOTAL = 510
EV_MAX_SINGLE = 252
LEVEL_100 = 100
LEVEL_50 = 50


def get_stat_defaults(gen: int) -> dict:
    """Return default IV/EV/nature values appropriate for a generation."""
    if gen <= 2:
        return {"iv": 15, "ev": 65535, "nature_mult": 1.0}
    return {"iv": 31, "ev": 85, "nature_mult": 1.0}


def calc_stat(base: int, iv: int, ev: int, level: int,
              nature_mult: float, is_hp: bool, gen: int = 9) -> int:
    """Calculate a Pokemon's actual stat value (gen-aware).

    Gen 1-2: DV system (0-15), Stat Exp (0-65535), no natures.
    Gen 3+: Modern IV (0-31), EV (0-255), natures.
    """
    if gen <= 2:
        # Gen 1-2 stat formula
        dv = min(iv, 15)
        stat_exp = min(ev, 65535)
        stat_exp_bonus = int(math.ceil(math.sqrt(max(stat_exp, 0)))) // 4
        if is_hp:
            if base == 1:  # Shedinja
                return 1
            return ((base + dv) * 2 + stat_exp_bonus) * level // 100 + level + 10
        return int((((base + dv) * 2 + stat_exp_bonus) * level // 100 + 5))
    else:
        # Gen 3+ stat formula
        if is_hp:
            if base == 1:  # Shedinja
                return 1
            return ((2 * base + iv + ev // 4) * level // 100) + level + 10
        return int((((2 * base + iv + ev // 4) * level // 100) + 5) * nature_mult)
