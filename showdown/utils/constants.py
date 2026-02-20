"""Constants for the Pokemon Showdown ML system."""

from enum import Enum
from typing import Dict, Tuple

TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]

TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}
NUM_TYPES = len(TYPES)

# Complete Gen 9 type effectiveness chart.
# Key: (attacking_type, defending_type) -> multiplier
# Only non-1.0 entries are stored; missing = 1.0 (neutral).
_TYPE_CHART_OVERRIDES: Dict[Tuple[str, str], float] = {
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

# Format types
FORMAT_SINGLES = "singles"
FORMAT_DOUBLES = "doubles"

# Map format prefixes to game type
FORMAT_GAME_TYPE = {
    "gen9ou": FORMAT_SINGLES,
    "gen9uu": FORMAT_SINGLES,
    "gen9ru": FORMAT_SINGLES,
    "gen9nu": FORMAT_SINGLES,
    "gen9pu": FORMAT_SINGLES,
    "gen9ubers": FORMAT_SINGLES,
    "gen9lc": FORMAT_SINGLES,
    "gen9monotype": FORMAT_SINGLES,
    "gen9vgc": FORMAT_DOUBLES,
    "gen9doublesou": FORMAT_DOUBLES,
    "gen9doublesuu": FORMAT_DOUBLES,
}


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


def calc_stat(base: int, iv: int, ev: int, level: int, nature_mult: float, is_hp: bool) -> int:
    """Calculate a Pokemon's actual stat value."""
    if is_hp:
        if base == 1:  # Shedinja
            return 1
        return ((2 * base + iv + ev // 4) * level // 100) + level + 10
    return int((((2 * base + iv + ev // 4) * level // 100) + 5) * nature_mult)
