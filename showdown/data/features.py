"""Feature extraction for ML models.

Three feature modes:
1. Embedding indices — for the neural network (species/move/item/ability as integer IDs)
2. Continuous features — per-Pokemon floats for the neural network (stats, types, moves, items)
3. Engineered features — for XGBoost (type coverage, stat distributions, synergy scores)
"""

import logging
import math
import re
from typing import Any

import numpy as np

from ..utils.constants import (
    TYPES, TYPE_TO_IDX, NUM_TYPES, STAT_NAMES,
    type_effectiveness, type_effectiveness_against, TEAM_SIZE,
    NATURES, calc_stat, IV_DEFAULT, LEVEL_100,
    extract_gen, get_type_chart_for_gen, get_stat_defaults,
    type_effectiveness_gen, type_effectiveness_against_gen,
    get_move_category, unify_special_stat,
)
from .mechanics import (
    ITEM_EFFECTS, ABILITY_EFFECTS,
    get_item_atk_mult, get_item_def_mult, get_item_damage_mult, get_item_spe_mult,
    get_ability_atk_mult, get_ability_stab_mult, get_ability_type_immunity,
    get_ability_bp_modifier, get_item_bp_modifier,
    get_move_secondary_chance, get_move_drain_fraction, get_move_recoil_fraction,
    get_move_boost_total,
    is_choice_item, is_offensive_item, is_defensive_item, is_recovery_item,
    is_berry, is_hazard_immune_item,
    ability_has_type_immunity, ability_sets_weather, ability_sets_terrain,
    ability_is_intimidate, ability_is_regenerator, ability_boosts_power,
    ability_boosts_speed, ability_ignores_abilities, ability_punishes_contact,
    ability_immune_to_status,
)
from .damage_calc import estimate_damage_pct, estimate_best_move_damage

log = logging.getLogger("showdown.data.features")


def _to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# Number of continuous features per Pokemon for the neural network
CONTINUOUS_DIM = 72


class FeatureExtractor:
    """Extract features from team data for both neural and XGBoost models."""

    # Backward-compat class-level constant
    CHOICE_ITEMS = {"choiceband", "choicespecs", "choicescarf"}

    def __init__(self, pokemon_data=None, gen: int = 9):
        """
        Args:
            pokemon_data: A loaded PokemonDataLoader instance. Required for
                         engineered features (type info, base stats).
            gen: Generation number (1-9). Controls type chart, stat formulas,
                 and normalization constants.
        """
        self.pokemon_data = pokemon_data
        self.gen = gen
        self._species_idx: dict[str, int] | None = None
        self._move_idx: dict[str, int] | None = None
        self._item_idx: dict[str, int] | None = None
        self._ability_idx: dict[str, int] | None = None

        # Per-generation type system
        gen_types, gen_type_to_idx, gen_chart, _ = get_type_chart_for_gen(gen)
        self._gen_types: list[str] = gen_types
        self._gen_type_to_idx: dict[str, int] = gen_type_to_idx
        self._gen_chart = gen_chart

        # Per-generation stat defaults
        self._stat_defaults = get_stat_defaults(gen)
        # Stat normalization ceiling (Gen 1-2 have lower stat ranges)
        self._stat_norm = 400.0 if gen <= 2 else 600.0

        self._init_move_sets()

    def _init_move_sets(self):
        """Initialize move category sets from loaded data or fallback defaults."""
        if (self.pokemon_data
                and hasattr(self.pokemon_data, "move_sets")
                and self.pokemon_data.move_sets):
            ms = self.pokemon_data.move_sets
            self.HAZARD_MOVES = ms.get("hazard", set())
            self.REMOVAL_MOVES = ms.get("removal", set())
            self.PRIORITY_MOVES = ms.get("priority", set())
            self.PIVOT_MOVES = ms.get("pivot", set())
            self.SETUP_MOVES = ms.get("setup", set())
            self.RECOVERY_MOVES = ms.get("recovery", set())
            self.DRAIN_MOVES = ms.get("drain", set())
            self.RECOIL_MOVES = ms.get("recoil", set())
            self.SPREAD_MOVES = ms.get("spread", set())
            self.CONTACT_MOVES = ms.get("contact", set())
            self.SOUND_MOVES = ms.get("sound", set())
            self.PHAZING_MOVES = ms.get("phazing", set())
            self.BURN_MOVES = ms.get("status_brn", set())
            self.PARA_MOVES = ms.get("status_par", set())
            self.SLEEP_MOVES = ms.get("status_slp", set())
            self.TOXIC_MOVES = ms.get("status_tox", set())
            self.POISON_MOVES = ms.get("status_psn", set())
            self.FREEZE_MOVES = ms.get("status_frz", set())
            self.STATUS_MOVES = (
                self.BURN_MOVES | self.PARA_MOVES | self.SLEEP_MOVES
                | self.TOXIC_MOVES | self.POISON_MOVES
            )
            log.info("Using data-driven move sets (%d categories)", len(ms))
        else:
            self.HAZARD_MOVES = {
                "stealthrock", "spikes", "toxicspikes", "stickyweb",
                "ceaselessedge", "stoneaxe",
            }
            self.REMOVAL_MOVES = {
                "rapidspin", "defog", "courtchange", "tidyup", "mortalspinx",
            }
            self.PRIORITY_MOVES = {
                "bulletpunch", "machpunch", "aquajet", "iceshard", "shadowsneak",
                "suckerpunch", "extremespeed", "fakeout", "quickattack",
                "accelerock", "jetpunch", "grassyglide", "firstimpression",
            }
            self.PIVOT_MOVES = {
                "uturn", "voltswitch", "flipturn", "partingshot",
                "teleport", "batonpass", "shedtail",
            }
            self.SETUP_MOVES = {
                "swordsdance", "nastyplot", "dragondance", "calmmind", "bulkup",
                "irondefense", "quiverdance", "shellsmash", "agility",
                "autotomize", "bellydrum", "coil", "curse", "geomancy",
                "growth", "tailglow", "shiftgear", "tidyup", "victorydance",
            }
            self.RECOVERY_MOVES = {
                "recover", "softboiled", "roost", "moonlight", "morningsun",
                "synthesis", "shoreup", "slackoff", "wish", "rest",
                "strengthsap",
            }
            self.DRAIN_MOVES = {
                "drainpunch", "gigadrain", "hornleech", "leechlife",
                "oblivionwing", "paraboliccharge", "drainingkiss", "strengthsap",
            }
            self.RECOIL_MOVES = {
                "flareblitz", "wildcharge", "bravebird", "headsmash",
                "doubleedge", "woodhammer", "volttackle", "headcharge",
                "takedown",
            }
            self.SPREAD_MOVES = {
                "earthquake", "surf", "rockslide", "heatwave", "dazzlinggleam",
                "discharge", "blizzard", "muddywater", "snarl", "hypervoice",
                "breakingswipe", "electroweb", "icywind", "bulldoze",
                "sludgewave", "eruption", "waterspout", "precipiceblades",
                "originpulse", "makeitrain",
            }
            self.CONTACT_MOVES = set()
            self.SOUND_MOVES = set()
            self.PHAZING_MOVES = {
                "roar", "whirlwind", "dragontail", "circlethrow",
            }
            self.BURN_MOVES = {
                "willowisp", "lavaplume", "scald", "scorchingsands",
                "inferno", "sacredfire",
            }
            self.PARA_MOVES = {
                "thunderwave", "stunspore", "nuzzle", "glare",
                "bodyslam", "discharge",
            }
            self.SLEEP_MOVES = {
                "spore", "sleeppowder", "hypnosis", "yawn", "darkvoid", "sing",
            }
            self.TOXIC_MOVES = {
                "toxic", "toxicspikes", "banefulbunker", "poisonfang",
                "malignantchain",
            }
            self.POISON_MOVES = set()
            self.FREEZE_MOVES = set()
            self.STATUS_MOVES = (
                self.BURN_MOVES | self.PARA_MOVES | self.SLEEP_MOVES
                | self.TOXIC_MOVES
            )

    # ------------------------------------------------------------------
    # Gen-aware type effectiveness helpers (instance methods)
    # ------------------------------------------------------------------

    def _type_eff(self, atk_type: str, def_type: str) -> float:
        """Single type-vs-type effectiveness using this instance's gen chart."""
        return self._gen_chart.get((atk_type, def_type), 1.0)

    def _type_eff_against(self, atk_type: str, def_types: list[str]) -> float:
        """Combined effectiveness against a Pokemon's type(s)."""
        mult = 1.0
        for dt in def_types:
            mult *= self._gen_chart.get((atk_type, dt), 1.0)
        return mult

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

    # ------------------------------------------------------------------
    # Continuous features per Pokemon (for neural network)
    # ------------------------------------------------------------------

    def pokemon_to_continuous(self, pkmn: dict) -> np.ndarray:
        """Extract 72 continuous features for a single Pokemon.

        [0-5]   Base stats (hp/atk/def/spa/spd/spe) / 255
        [6-11]  Computed stats using EVs/nature / 600
        [12-13] Type indices (type1/17, type2/17 or -1)
        [14-31] Defensive profile: log2(eff) for each of 18 attacking types
        [32-35] Move quality: best phys bp, best spec bp, STAB count, has priority
        [36-39] Item category: choice, life orb, boots/leftovers, sash/berry
        [40-41] Ability: grants immunity, boosts offense/speed
        [42-47] Item numeric multipliers: atk/spa/damage/def/spd/spe
        [48-54] Ability features: atk_mult, stab_mult, immunity, weather, terrain, intimidate, regen
        [55-59] Move detail: best secondary %, total drain, total recoil, contact frac, boost potential
        [60]    Has phazing move
        [61-63] Effective power/bulk: eff phys, eff spec, eff bulk
        [64]    Mega Stone held (Gen 6-7)
        [65]    Z-Crystal held (Gen 7)
        [66]    Best Z-Move base power / 200 (Gen 7)
        [67]    Can Dynamax (Gen 8)
        [68]    Best Max Move power / 150 (Gen 8)
        [69]    Tera type index / 17 (Gen 9, -1 if none)
        [70]    Tera adds new STAB (Gen 9)
        [71]    Tera weakness reduction (Gen 9)
        """
        feats = np.zeros(CONTINUOUS_DIM, dtype=np.float32)
        species_id = _to_id(pkmn.get("species", ""))

        # Base stats [0-5]
        base_stats = self._get_base_stats(species_id)
        if base_stats:
            base_stats = unify_special_stat(base_stats, self.gen)
            for i, sn in enumerate(STAT_NAMES):
                feats[i] = base_stats.get(sn, 0) / 255.0

        # Computed stats [6-11]
        computed = None
        if base_stats:
            computed = self._compute_actual_stats(pkmn, base_stats)
            for i, sn in enumerate(STAT_NAMES):
                feats[6 + i] = computed.get(sn, 0) / self._stat_norm

        # Type encoding [12-13]
        types = self._get_types(species_id)
        if len(types) >= 1:
            feats[12] = TYPE_TO_IDX.get(types[0], 0) / 17.0
        if len(types) >= 2:
            feats[13] = TYPE_TO_IDX.get(types[1], 0) / 17.0
        else:
            feats[13] = -1.0

        # Defensive profile [14-31]
        if types:
            for atk_idx, atk_type in enumerate(TYPES):
                eff = self._type_eff_against(atk_type, types)
                if eff == 0:
                    feats[14 + atk_idx] = -4.0
                else:
                    feats[14 + atk_idx] = math.log2(eff)

        # Move quality [32-35]
        best_phys_bp = 0
        best_spec_bp = 0
        stab_count = 0
        has_priority = 0
        best_secondary = 0.0
        total_drain = 0.0
        total_recoil = 0.0
        contact_count = 0
        damaging_count = 0
        boost_potential = 0
        has_phazing = 0
        for move_name in pkmn.get("moves", []):
            if not move_name:
                continue
            mid = _to_id(move_name)
            if mid in self.PRIORITY_MOVES:
                has_priority = 1
            if mid in self.PHAZING_MOVES:
                has_phazing = 1
            move_data = self._get_move(move_name)
            if move_data:
                bp = move_data.get("basePower", 0)
                cat = get_move_category(move_data, self.gen)
                mtype = move_data.get("type", "")
                if cat != "Status" and bp > 0:
                    damaging_count += 1
                    if cat == "Physical" and bp > best_phys_bp:
                        best_phys_bp = bp
                    elif cat == "Special" and bp > best_spec_bp:
                        best_spec_bp = bp
                    if mtype in types:
                        stab_count += 1
                    if move_data.get("flags", {}).get("contact"):
                        contact_count += 1
                # Extract move detail features from JSON
                sec_chance = get_move_secondary_chance(move_data)
                if sec_chance > best_secondary:
                    best_secondary = sec_chance
                total_drain += get_move_drain_fraction(move_data)
                total_recoil += get_move_recoil_fraction(move_data)
                boost_potential += max(get_move_boost_total(move_data), 0)

        feats[32] = best_phys_bp / 150.0
        feats[33] = best_spec_bp / 150.0
        feats[34] = stab_count / 4.0
        feats[35] = float(has_priority)

        # Item category [36-39]
        item_id = _to_id(pkmn.get("item", "") or "")
        feats[36] = 1.0 if is_choice_item(item_id) else 0.0
        feats[37] = 1.0 if item_id == "lifeorb" else 0.0
        feats[38] = 1.0 if item_id in ("heavydutyboots", "leftovers", "blacksludge") else 0.0
        feats[39] = 1.0 if item_id in ("focussash", "focusband") or is_berry(item_id) else 0.0

        # Ability category [40-41]
        ability_id = _to_id(pkmn.get("ability", "") or "")
        feats[40] = 1.0 if ability_has_type_immunity(ability_id) else 0.0
        feats[41] = 1.0 if ability_boosts_power(ability_id) or ability_boosts_speed(ability_id) else 0.0

        # --- NEW: Item numeric multipliers [42-47] ---
        feats[42] = get_item_atk_mult(item_id, True) / 2.0   # phys atk mult
        feats[43] = get_item_atk_mult(item_id, False) / 2.0  # spec atk mult
        feats[44] = get_item_damage_mult(item_id) / 2.0
        feats[45] = get_item_def_mult(item_id, True) / 2.0   # phys def mult
        feats[46] = get_item_def_mult(item_id, False) / 2.0  # spec def mult
        feats[47] = get_item_spe_mult(item_id) / 2.0

        # --- NEW: Ability features [48-54] ---
        feats[48] = get_ability_atk_mult(ability_id) / 2.0
        feats[49] = get_ability_stab_mult(ability_id) / 2.0
        feats[50] = 1.0 if ability_has_type_immunity(ability_id) else 0.0
        feats[51] = 1.0 if ability_sets_weather(ability_id) else 0.0
        feats[52] = 1.0 if ability_sets_terrain(ability_id) else 0.0
        feats[53] = 1.0 if ability_is_intimidate(ability_id) else 0.0
        feats[54] = 1.0 if ability_is_regenerator(ability_id) else 0.0

        # --- NEW: Move detail features [55-60] ---
        feats[55] = best_secondary / 100.0
        feats[56] = min(total_drain, 1.0)
        feats[57] = min(total_recoil, 1.0)
        feats[58] = contact_count / max(damaging_count, 1)
        feats[59] = min(boost_potential / 6.0, 1.0)
        feats[60] = float(has_phazing)

        # --- NEW: Effective power/bulk [61-63] ---
        if base_stats and computed:
            atk_stat = computed.get("atk", 80)
            spa_stat = computed.get("spa", 80)
            def_stat = computed.get("def", 80)
            spd_stat = computed.get("spd", 80)
            hp_stat = computed.get("hp", 200)

            # Effective power = best BP * stat * item_mult * ability_mult
            phys_mult = get_item_atk_mult(item_id, True) * get_ability_atk_mult(ability_id)
            spec_mult = get_item_atk_mult(item_id, False) * get_ability_atk_mult(ability_id)
            feats[61] = (best_phys_bp * atk_stat * phys_mult) / 60000.0
            feats[62] = (best_spec_bp * spa_stat * spec_mult) / 60000.0

            # Effective bulk = HP * avg(Def * def_mult, SpD * spd_mult)
            eff_def = def_stat * get_item_def_mult(item_id, True)
            eff_spd = spd_stat * get_item_def_mult(item_id, False)
            feats[63] = (hp_stat * (eff_def + eff_spd) / 2) / 150000.0

        # === Generation-specific gimmick features [64-71] ===
        from .mechanics import is_mega_stone, is_z_crystal, get_z_crystal_type, get_z_move_bp, get_max_move_bp

        # [64] Mega Stone (Gen 6-7): Pokemon holds a Mega Stone
        if self.gen in (6, 7) and is_mega_stone(item_id):
            feats[64] = 1.0

        # [65-66] Z-Crystal (Gen 7): holds Z-Crystal + best Z-Move power
        if self.gen == 7 and is_z_crystal(item_id):
            feats[65] = 1.0
            z_type = get_z_crystal_type(item_id)
            if z_type:
                best_z_bp = 0
                for move_name in pkmn.get("moves", []):
                    if not move_name:
                        continue
                    mid = _to_id(move_name)
                    mdata = self._get_move(mid) or {}
                    if mdata.get("category") in ("Physical", "Special"):
                        m_type = mdata.get("type", "")
                        m_bp = mdata.get("basePower", 0) or 0
                        # Type Z-Crystals boost same-type moves
                        if m_type == z_type and m_bp > 0:
                            best_z_bp = max(best_z_bp, get_z_move_bp(m_bp))
                feats[66] = best_z_bp / 200.0

        # [67-68] Dynamax (Gen 8): best Max Move power
        if self.gen == 8:
            feats[67] = 1.0  # can_dynamax (always true in Gen 8)
            best_max_bp = 0
            for move_name in pkmn.get("moves", []):
                if not move_name:
                    continue
                mid = _to_id(move_name)
                mdata = self._get_move(mid) or {}
                if mdata.get("category") in ("Physical", "Special"):
                    m_bp = mdata.get("basePower", 0) or 0
                    m_type = mdata.get("type", "")
                    if m_bp > 0:
                        best_max_bp = max(best_max_bp, get_max_move_bp(m_bp, m_type))
            feats[68] = best_max_bp / 150.0

        # [69-71] Tera Type (Gen 9): type index + STAB gain
        if self.gen >= 9:
            tera_type = pkmn.get("tera_type")
            if tera_type:
                tera_id = _to_id(tera_type)
                tera_idx = TYPE_TO_IDX.get(tera_type, TYPE_TO_IDX.get(tera_id.capitalize(), -1))
                feats[69] = tera_idx / 17.0 if tera_idx >= 0 else -1.0
                # Does Tera add STAB coverage the Pokemon doesn't already have?
                types = self._get_types(species_id)
                if tera_type not in types:
                    feats[70] = 1.0  # Tera adds new STAB type
                # Tera defensive benefit: does it remove a weakness?
                if types:
                    # Count weaknesses before and after Tera
                    weak_before = sum(1 for t in TYPES if self._type_eff_against(t, types) > 1.0)
                    weak_after = sum(1 for t in TYPES if self._type_eff_against(t, [tera_type]) > 1.0)
                    if weak_after < weak_before:
                        feats[71] = (weak_before - weak_after) / 7.0  # normalize

        return feats

    def team_to_continuous(self, team: list[dict]) -> np.ndarray:
        """Extract continuous features for a full team.

        Returns: np.ndarray of shape (6, CONTINUOUS_DIM)
        """
        result = np.zeros((TEAM_SIZE, CONTINUOUS_DIM), dtype=np.float32)
        for i, pkmn in enumerate(team[:TEAM_SIZE]):
            result[i] = self.pokemon_to_continuous(pkmn)
        return result

    def _compute_actual_stats(self, pkmn: dict, base_stats: dict) -> dict[str, int]:
        """Compute actual stat values using EVs/nature (inferred if absent)."""
        nature_name = pkmn.get("nature")
        evs = pkmn.get("evs", {})
        ivs = pkmn.get("ivs", {})

        # Gen 1-2: no natures, different IV/EV defaults
        if self.gen <= 2:
            nature_name = "Hardy"  # no natures in Gen 1-2
            if not ivs:
                ivs = {sn: self._stat_defaults["iv"] for sn in STAT_NAMES}
            if not evs or not any(v > 0 for v in evs.values()):
                evs = {sn: self._stat_defaults["ev"] for sn in STAT_NAMES}
        else:
            if not nature_name or not evs or not any(v > 0 for v in evs.values()):
                spread = self._infer_spread(pkmn)
                nature_name = nature_name or spread.get("nature", "Hardy")
                if not evs or not any(v > 0 for v in evs.values()):
                    evs = spread.get("evs", {})
                if not ivs:
                    ivs = spread.get("ivs", {})

        nature_mults = NATURES.get(nature_name, {})
        base_stats = unify_special_stat(base_stats, self.gen)
        computed = {}
        for sn in STAT_NAMES:
            base = base_stats.get(sn, 80)
            iv = ivs.get(sn, self._stat_defaults["iv"])
            ev = evs.get(sn, 0)
            nat_mult = nature_mults.get(sn, 1.0)
            computed[sn] = calc_stat(base, iv, ev, LEVEL_100, nat_mult, sn == "hp", self.gen)
        return computed

    def _infer_spread(self, pkmn: dict) -> dict:
        """Infer EV spread and nature for a Pokemon based on its set."""
        try:
            from ..teambuilder.spread_inference import infer_spread
            if self.pokemon_data:
                return infer_spread(pkmn, self.pokemon_data)
        except (ImportError, Exception):
            pass
        return {"nature": "Hardy", "evs": {}, "ivs": {}}

    def battle_to_tensors(self, battle: dict) -> dict[str, Any]:
        """Convert a full battle record into tensor-ready data.

        Returns dict with team1_*, team2_* index arrays, continuous features,
        rating info (6 features), and label.
        """
        t1 = self.team_to_indices(battle["team1"])
        t2 = self.team_to_indices(battle["team2"])
        label = 1.0 if battle["winner"] == 1 else 0.0

        # Continuous per-Pokemon features (6, 42) for each team
        t1_cont = self.team_to_continuous(battle["team1"])
        t2_cont = self.team_to_continuous(battle["team2"])

        # Rating features (6 dimensions) — both players' pre-battle Elo
        r1 = battle.get("rating1") or 0
        r2 = battle.get("rating2") or 0
        has_both = r1 > 0 and r2 > 0
        r1_safe = r1 if r1 > 0 else 1500
        r2_safe = r2 if r2 > 0 else 1500
        rating_features = np.array([
            r1_safe / 2000.0,                      # P1 normalized rating
            r2_safe / 2000.0,                      # P2 normalized rating
            (r1_safe - r2_safe) / 400.0,           # rating difference (key predictor)
            float(has_both),                       # data quality flag
            max(r1_safe, r2_safe) / 2000.0,        # game quality (higher-rated player)
            abs(r1_safe - r2_safe) / 400.0,        # matchup lopsidedness
        ], dtype=np.float32)

        return {
            "team1_species": t1["species"],
            "team1_moves": t1["moves"],
            "team1_items": t1["items"],
            "team1_abilities": t1["abilities"],
            "team2_species": t2["species"],
            "team2_moves": t2["moves"],
            "team2_items": t2["items"],
            "team2_abilities": t2["abilities"],
            "team1_continuous": t1_cont,
            "team2_continuous": t2_cont,
            "rating_features": rating_features,
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
                stats = unify_special_stat(stats, self.gen)
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
                        eff = self._type_eff_against(move_type, [def_type])
                        if eff >= 2.0:
                            off_coverage[def_idx] = 1.0
        features.extend(off_coverage.tolist())

        # Defensive coverage (18)
        def_coverage = np.zeros(NUM_TYPES, dtype=np.float32)
        for atk_idx, atk_type in enumerate(TYPES):
            best_resist = 4.0
            for types in info["types_list"]:
                if types:
                    eff = self._type_eff_against(atk_type, types)
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

        # ---- NEW Phase 1B feature groups ----

        # Move quality distribution (20)
        features.extend(self._move_quality_features(team[:TEAM_SIZE]))

        # Computed stat features (12)
        features.extend(self._computed_stat_features(team[:TEAM_SIZE]))

        # Item effect features (12)
        features.extend(self._item_effect_features(team[:TEAM_SIZE]))

        # Ability effect features (12)
        features.extend(self._ability_effect_features(team[:TEAM_SIZE]))

        # Type synergy features (6)
        features.extend(self._type_synergy_features(team[:TEAM_SIZE]))

        # ---- Phase 2 feature groups ----

        # Move interaction features (10)
        features.extend(self._move_interaction_features(team[:TEAM_SIZE]))

        # Effective power features (8)
        features.extend(self._effective_power_features(team[:TEAM_SIZE]))

        # Enhanced item/ability numeric features (8)
        features.extend(self._enhanced_item_ability_numeric_features(team[:TEAM_SIZE]))

        # Detailed status features (6)
        features.extend(self._detailed_status_features(team[:TEAM_SIZE]))

        # ---- Generation gimmick features (8) ----
        features.extend(self._gimmick_features(team[:TEAM_SIZE]))

        return np.array(features, dtype=np.float32)

    def battle_to_engineered(self, battle: dict) -> tuple[np.ndarray, float]:
        """Extract XGBoost features for a battle.

        Includes per-team features + diff + matchup interactions + rating features.
        Returns (features, label).
        """
        t1 = battle["team1"][:TEAM_SIZE]
        t2 = battle["team2"][:TEAM_SIZE]

        t1_feat = self.team_to_engineered(t1)
        t2_feat = self.team_to_engineered(t2)
        diff = t1_feat - t2_feat

        # Matchup interaction features
        matchup_feat = self._matchup_features(t1, t2)

        # Rating features for XGBoost (6) — both players' pre-battle Elo
        r1 = battle.get("rating1") or 0
        r2 = battle.get("rating2") or 0
        has_both = r1 > 0 and r2 > 0
        r1_safe = r1 if r1 > 0 else 1500
        r2_safe = r2 if r2 > 0 else 1500
        rating_feat = np.array([
            r1_safe / 2000.0,                      # P1 normalized rating
            r2_safe / 2000.0,                      # P2 normalized rating
            (r1_safe - r2_safe) / 400.0,           # rating difference (key predictor)
            float(has_both),                       # data quality flag
            max(r1_safe, r2_safe) / 2000.0,        # game quality
            abs(r1_safe - r2_safe) / 400.0,        # matchup lopsidedness
        ], dtype=np.float32)

        features = np.concatenate([t1_feat, t2_feat, diff, matchup_feat, rating_feat])
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
                        eff = self._type_eff_against(move_data.get("type", ""), types2)
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
                        eff = self._type_eff_against(move_data.get("type", ""), types1)
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
                        eff = self._type_eff_against(stab_type, types1)
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
                        eff = self._type_eff_against(stab_type, types2)
                        if eff < 1.0:
                            resist_score_rev += 1
                            break
        features.append(resist_score_rev / max(resist_total_rev, 1))

        # BST advantage
        bst1 = sum(sum(s.get(sn, 0) for sn in STAT_NAMES) for s in speeds1) / max(len(team1), 1)
        bst2 = sum(sum(s.get(sn, 0) for sn in STAT_NAMES) for s in speeds2) / max(len(team2), 1)
        features.append((bst1 - bst2) / 720.0)

        # ---- NEW expanded matchup features ----

        # Threat matrix: estimate best move damage % for each attacker-defender pair
        threat_t1 = []  # t1 attacking t2
        threat_t2 = []  # t2 attacking t1
        for p1 in team1:
            for p2 in team2:
                threat_t1.append(self._estimate_threat(p1, p2))
        for p2 in team2:
            for p1 in team1:
                threat_t2.append(self._estimate_threat(p2, p1))

        # Threat matrix stats (6): mean/max/min for each direction
        if threat_t1:
            features.extend([np.mean(threat_t1), max(threat_t1), min(threat_t1)])
        else:
            features.extend([0.0, 0.0, 0.0])
        if threat_t2:
            features.extend([np.mean(threat_t2), max(threat_t2), min(threat_t2)])
        else:
            features.extend([0.0, 0.0, 0.0])

        # 2HKO potential: fraction of matchups where attacker can 2HKO (2)
        t1_2hko = sum(1 for t in threat_t1 if t > 0.45) / max(len(threat_t1), 1)
        t2_2hko = sum(1 for t in threat_t2 if t > 0.45) / max(len(threat_t2), 1)
        features.extend([t1_2hko, t2_2hko])

        # Safe switch-in count: defenders that take < 25% from any attacker (2)
        t1_safe = 0
        for j, p2 in enumerate(team2):
            worst = max(threat_t1[i * len(team2) + j] for i in range(len(team1))) if team1 else 0
            if worst < 0.25:
                t1_safe += 1
        t2_safe = 0
        for j, p1 in enumerate(team1):
            worst = max(threat_t2[i * len(team1) + j] for i in range(len(team2))) if team2 else 0
            if worst < 0.25:
                t2_safe += 1
        features.extend([t1_safe / max(len(team2), 1), t2_safe / max(len(team1), 1)])

        # Stealth Rock damage comparison (2)
        sr_dmg_t1 = sum(self._sr_damage_fraction(p) for p in team1)
        sr_dmg_t2 = sum(self._sr_damage_fraction(p) for p in team2)
        features.extend([sr_dmg_t1 / max(len(team1), 1), sr_dmg_t2 / max(len(team2), 1)])

        # STAB immunity count: how many opponents are immune to a team's STAB (2)
        t1_immune = 0
        for p1 in team1:
            types1 = self._get_types(_to_id(p1.get("species", "")))
            for stab_type in types1:
                for p2 in team2:
                    types2 = self._get_types(_to_id(p2.get("species", "")))
                    if types2 and self._type_eff_against(stab_type, types2) == 0:
                        t1_immune += 1
                        break
        t2_immune = 0
        for p2 in team2:
            types2 = self._get_types(_to_id(p2.get("species", "")))
            for stab_type in types2:
                for p1 in team1:
                    types1 = self._get_types(_to_id(p1.get("species", "")))
                    if types1 and self._type_eff_against(stab_type, types1) == 0:
                        t2_immune += 1
                        break
        features.extend([t1_immune / max(len(team1) * 2, 1), t2_immune / max(len(team2) * 2, 1)])

        # Priority kill potential: team has priority AND high attack (2)
        t1_prio_threat = 0
        t2_prio_threat = 0
        for p in team1:
            moves_set = {_to_id(m) for m in p.get("moves", []) if m}
            if moves_set & self.PRIORITY_MOVES:
                base = self._get_base_stats(_to_id(p.get("species", ""))) or {}
                if base.get("atk", 0) >= 100 or base.get("spa", 0) >= 100:
                    t1_prio_threat += 1
        for p in team2:
            moves_set = {_to_id(m) for m in p.get("moves", []) if m}
            if moves_set & self.PRIORITY_MOVES:
                base = self._get_base_stats(_to_id(p.get("species", ""))) or {}
                if base.get("atk", 0) >= 100 or base.get("spa", 0) >= 100:
                    t2_prio_threat += 1
        features.extend([t1_prio_threat / 6.0, t2_prio_threat / 6.0])

        # ---- Enhanced matchup features (8 new) ----
        n1, n2 = len(team1), len(team2)

        # Ability immunity denial (2)
        t1_denied = 0
        t2_denied = 0
        for p2 in team2:
            def_ab_id = _to_id(p2.get("ability", "") or "")
            immune_type = get_ability_type_immunity(def_ab_id)
            if immune_type:
                for p1 in team1:
                    for m in p1.get("moves", []):
                        md = self._get_move(m)
                        if md and md.get("type") == immune_type and md.get("basePower", 0) > 0:
                            t1_denied += 1
                            break
        for p1 in team1:
            atk_ab_id = _to_id(p1.get("ability", "") or "")
            immune_type = get_ability_type_immunity(atk_ab_id)
            if immune_type:
                for p2 in team2:
                    for m in p2.get("moves", []):
                        md = self._get_move(m)
                        if md and md.get("type") == immune_type and md.get("basePower", 0) > 0:
                            t2_denied += 1
                            break
        features.extend([t1_denied / max(n1 * n2, 1), t2_denied / max(n1 * n2, 1)])

        # Intimidate impact (2)
        t1_intim = sum(1 for p in team1 if ability_is_intimidate(_to_id(p.get("ability", "") or "")))
        t2_intim = sum(1 for p in team2 if ability_is_intimidate(_to_id(p.get("ability", "") or "")))
        t2_phys_count = sum(1 for p in team2
                           if (self._get_base_stats(_to_id(p.get("species", ""))) or {}).get("atk", 0) >
                              (self._get_base_stats(_to_id(p.get("species", ""))) or {}).get("spa", 0))
        t1_phys_count = sum(1 for p in team1
                           if (self._get_base_stats(_to_id(p.get("species", ""))) or {}).get("atk", 0) >
                              (self._get_base_stats(_to_id(p.get("species", ""))) or {}).get("spa", 0))
        features.append(t1_intim * t2_phys_count / max(n2, 1) / 6.0)
        features.append(t2_intim * t1_phys_count / max(n1, 1) / 6.0)

        # Entry hazard vs non-Boots ratio (2)
        t1_boots = sum(1 for p in team1 if is_hazard_immune_item(_to_id(p.get("item", "") or "")))
        t2_boots = sum(1 for p in team2 if is_hazard_immune_item(_to_id(p.get("item", "") or "")))
        t1_haz = any(_to_id(m) in self.HAZARD_MOVES for p in team1 for m in p.get("moves", []) if m)
        t2_haz = any(_to_id(m) in self.HAZARD_MOVES for p in team2 for m in p.get("moves", []) if m)
        features.append((n2 - t2_boots) / max(n2, 1) if t1_haz else 0.0)
        features.append((n1 - t1_boots) / max(n1, 1) if t2_haz else 0.0)

        # Regen/pivot sustainability (2)
        t1_rp = sum(1 for p in team1
                    if ability_is_regenerator(_to_id(p.get("ability", "") or ""))
                    and {_to_id(m) for m in p.get("moves", []) if m} & self.PIVOT_MOVES)
        t2_rp = sum(1 for p in team2
                    if ability_is_regenerator(_to_id(p.get("ability", "") or ""))
                    and {_to_id(m) for m in p.get("moves", []) if m} & self.PIVOT_MOVES)
        features.extend([t1_rp / 6.0, t2_rp / 6.0])

        return np.array(features, dtype=np.float32)

    def _estimate_threat(self, attacker: dict, defender: dict) -> float:
        """Estimate best move damage as fraction of defender HP (0-2 range).

        Uses the full competitive damage calculator with item/ability modifiers.
        """
        atk_id = _to_id(attacker.get("species", ""))
        def_id = _to_id(defender.get("species", ""))
        atk_base = unify_special_stat(self._get_base_stats(atk_id) or {}, self.gen)
        def_base = unify_special_stat(self._get_base_stats(def_id) or {}, self.gen)
        atk_types = self._get_types(atk_id)
        def_types = self._get_types(def_id)

        item_id = _to_id(attacker.get("item", "") or "")
        ability_id = _to_id(attacker.get("ability", "") or "")
        def_item_id = _to_id(defender.get("item", "") or "")
        def_ability_id = _to_id(defender.get("ability", "") or "")

        hp_val = def_base.get("hp", 80)
        # Build pre-digested dicts for damage_calc
        atk_data = {
            "atk": atk_base.get("atk", 80),
            "spa": atk_base.get("spa", 80),
            "def": atk_base.get("def", 80),
            "types": atk_types,
            "stab_types": set(atk_types),
            "item_id": item_id,
            "ability_id": ability_id,
            "item_effects": ITEM_EFFECTS.get(item_id, {}),
            "ability_effects": ABILITY_EFFECTS.get(ability_id, {}),
            "atk_item_mult": get_item_atk_mult(item_id, True),
            "spa_item_mult": get_item_atk_mult(item_id, False),
            "damage_mult": get_item_damage_mult(item_id),
            "stab_mult": get_ability_stab_mult(ability_id),
        }
        def_data = {
            "def": max(def_base.get("def", 80), 1),
            "spd": max(def_base.get("spd", 80), 1),
            "hp_actual": max(calc_stat(hp_val, self._stat_defaults["iv"],
                                       self._stat_defaults["ev"], LEVEL_100,
                                       1.0, True, self.gen), 1),
            "types": def_types,
            "item_id": def_item_id,
            "ability_id": def_ability_id,
            "item_effects": ITEM_EFFECTS.get(def_item_id, {}),
            "ability_effects": ABILITY_EFFECTS.get(def_ability_id, {}),
            "def_item_mult": get_item_def_mult(def_item_id, True),
            "spd_item_mult": get_item_def_mult(def_item_id, False),
        }

        best = 0.0
        for move_name in attacker.get("moves", []):
            if not move_name:
                continue
            move_data = self._get_move(move_name)
            if not move_data:
                continue
            dmg = estimate_damage_pct(atk_data, def_data, move_data, type_chart=self._gen_chart, gen=self.gen)
            if dmg > best:
                best = dmg
        return best

    def _sr_damage_fraction(self, pkmn: dict) -> float:
        """Calculate Stealth Rock damage as fraction of HP for a Pokemon."""
        species_id = _to_id(pkmn.get("species", ""))
        types = self._get_types(species_id)
        if not types:
            return 0.125
        eff = self._type_eff_against("Rock", types)
        return min(eff * 0.125, 0.5)  # cap at 50%

    # ------------------------------------------------------------------
    # New XGBoost team feature groups (Phase 1B)
    # ------------------------------------------------------------------

    def _move_quality_features(self, team: list[dict]) -> list[float]:
        """Extract move quality distribution features (20 features).

        Uses data-driven move sets and JSON fields instead of hardcoded sets.
        """
        all_bps = []
        stab_count = 0
        priority_count = 0
        high_power_count = 0
        burn_count = 0
        para_count = 0
        sleep_count = 0
        toxic_count = 0
        secondary_count = 0
        pivot_count = 0
        setup_count = 0
        recovery_count = 0
        attacking_types = set()
        accuracies = []
        spread_count = 0
        drain_count = 0
        recoil_count = 0

        for pkmn in team:
            species_id = _to_id(pkmn.get("species", ""))
            pkmn_types = self._get_types(species_id)
            pkmn_moves = set()

            for move_name in pkmn.get("moves", []):
                if not move_name:
                    continue
                mid = _to_id(move_name)
                pkmn_moves.add(mid)
                move_data = self._get_move(move_name)

                if move_data:
                    bp = move_data.get("basePower", 0)
                    cat = move_data.get("category", "")
                    mtype = move_data.get("type", "")

                    if cat != "Status" and bp > 0:
                        all_bps.append(bp)
                        if bp >= 100:
                            high_power_count += 1
                        if mtype in pkmn_types:
                            stab_count += 1
                        attacking_types.add(mtype)

                    acc = move_data.get("accuracy")
                    if acc is True:
                        accuracies.append(100.0)
                    elif isinstance(acc, (int, float)) and acc > 0:
                        accuracies.append(float(acc))

                    # Use JSON priority field
                    if move_data.get("priority", 0) > 0:
                        priority_count += 1

                    # Use JSON secondary/secondaries fields
                    if move_data.get("secondary") or move_data.get("secondaries"):
                        secondary_count += 1

                    # Use JSON drain/recoil fields
                    if move_data.get("drain"):
                        drain_count += 1
                    if move_data.get("recoil") or move_data.get("self", {}).get("boosts"):
                        # Recoil + self-stat-drop moves (Close Combat, etc.)
                        if move_data.get("recoil"):
                            recoil_count += 1

                    # Use JSON target field for spread moves
                    target = move_data.get("target", "")
                    if target in ("allAdjacentFoes", "allAdjacent", "all"):
                        spread_count += 1

                # Status via data-driven sets
                if mid in self.BURN_MOVES:
                    burn_count += 1
                if mid in self.PARA_MOVES:
                    para_count += 1
                if mid in self.SLEEP_MOVES:
                    sleep_count += 1
                if mid in self.TOXIC_MOVES:
                    toxic_count += 1

            if pkmn_moves & self.PIVOT_MOVES:
                pivot_count += 1
            if pkmn_moves & self.SETUP_MOVES:
                setup_count += 1
            if pkmn_moves & self.RECOVERY_MOVES:
                recovery_count += 1

        total_bp = sum(all_bps)
        mean_bp = float(np.mean(all_bps)) if all_bps else 0.0
        max_bp = max(all_bps) if all_bps else 0.0
        std_bp = float(np.std(all_bps)) if len(all_bps) > 1 else 0.0
        coverage = len(attacking_types) / NUM_TYPES
        avg_acc = float(np.mean(accuracies)) if accuracies else 100.0

        return [
            total_bp / 1000.0, mean_bp / 150.0, max_bp / 200.0, std_bp / 50.0,
            stab_count / 12.0, priority_count / 6.0, high_power_count / 12.0,
            burn_count / 6.0, para_count / 6.0, sleep_count / 6.0,
            toxic_count / 6.0, secondary_count / 12.0,
            pivot_count / 6.0, setup_count / 6.0, recovery_count / 6.0,
            coverage, avg_acc / 100.0,
            spread_count / 12.0, drain_count / 6.0, recoil_count / 6.0,
        ]

    def _computed_stat_features(self, team: list[dict]) -> list[float]:
        """Extract computed stat features using actual EVs/nature (12 features)."""
        speeds = []
        hps = []
        atks = []
        spas = []
        defs_list = []
        spds = []

        for pkmn in team:
            species_id = _to_id(pkmn.get("species", ""))
            base = self._get_base_stats(species_id)
            if not base:
                continue
            computed = self._compute_actual_stats(pkmn, base)
            speeds.append(computed.get("spe", 0))
            hps.append(computed.get("hp", 0))
            atks.append(computed.get("atk", 0))
            spas.append(computed.get("spa", 0))
            defs_list.append(computed.get("def", 0))
            spds.append(computed.get("spd", 0))

        if not speeds:
            return [0.0] * 12

        fast_count = sum(1 for s in speeds if s > 300)
        strong_count = sum(1 for a in atks + spas if a > 350)
        phys_bulk = sum(h * d for h, d in zip(hps, defs_list)) / len(hps)
        spec_bulk = sum(h * s for h, s in zip(hps, spds)) / len(hps)

        return [
            float(np.mean(speeds)) / 400.0,
            max(speeds) / 500.0,
            float(np.mean(hps)) / 400.0,
            float(np.mean(atks)) / 400.0,
            float(np.mean(spas)) / 400.0,
            float(np.mean(defs_list)) / 400.0,
            float(np.mean(spds)) / 400.0,
            fast_count / 6.0,
            strong_count / 6.0,
            phys_bulk / 100000.0,
            spec_bulk / 100000.0,
            (phys_bulk + spec_bulk) / 200000.0,
        ]

    def _item_effect_features(self, team: list[dict]) -> list[float]:
        """Count item categories across the team (12 features).

        Uses mechanics registry instead of hardcoded sets.
        """
        c = [0] * 12
        for pkmn in team:
            iid = _to_id(pkmn.get("item", "") or "")
            if is_choice_item(iid):              c[0] += 1
            if iid == "lifeorb":                 c[1] += 1
            if iid in ("heavydutyboots", "leftovers", "blacksludge"): c[2] += 1
            if is_hazard_immune_item(iid):       c[3] += 1
            if iid in ("focussash", "focusband") or is_berry(iid): c[4] += 1
            if is_offensive_item(iid):           c[5] += 1
            if is_defensive_item(iid):           c[6] += 1
            if iid == "assaultvest":             c[7] += 1
            if iid == "eviolite":                c[8] += 1
            if is_berry(iid):                    c[9] += 1
            eff = ITEM_EFFECTS.get(iid, {})
            if eff.get("category") == "terrain_seed": c[10] += 1
            if iid == "rockyhelmet":             c[11] += 1
        return [v / 6.0 for v in c]

    def _ability_effect_features(self, team: list[dict]) -> list[float]:
        """Count ability categories across the team (12 features).

        Uses mechanics registry instead of hardcoded sets.
        """
        c = [0] * 12
        for pkmn in team:
            aid = _to_id(pkmn.get("ability", "") or "")
            if ability_has_type_immunity(aid):    c[0] += 1
            if ability_sets_weather(aid):         c[1] += 1
            if ability_sets_terrain(aid):         c[2] += 1
            if ability_is_intimidate(aid):        c[3] += 1
            if ability_boosts_speed(aid):         c[4] += 1
            if ability_boosts_power(aid):         c[5] += 1
            eff = ABILITY_EFFECTS.get(aid, {})
            if eff.get("indirect_damage_immune") or eff.get("reflects_status"): c[6] += 1
            if ability_is_regenerator(aid):       c[7] += 1
            if ability_immune_to_status(aid):     c[8] += 1
            if eff.get("status_priority") or eff.get("flying_priority") or eff.get("heal_priority"): c[9] += 1
            if ability_punishes_contact(aid):     c[10] += 1
            if ability_ignores_abilities(aid):    c[11] += 1
        return [v / 6.0 for v in c]

    def _type_synergy_features(self, team: list[dict]) -> list[float]:
        """Extract type synergy features (6 features)."""
        all_types = []
        for pkmn in team:
            types = self._get_types(_to_id(pkmn.get("species", "")))
            all_types.extend(types)

        unique_types = set(all_types)
        unique_count = len(unique_types)
        redundancy = 1.0 - (unique_count / max(len(all_types), 1))

        # Worst weakness: max number of team members hit SE by one type
        worst = 0
        unresisted = 0
        doubly_resisted = 0
        for atk_type in TYPES:
            se_count = 0
            resisted = False
            for pkmn in team:
                types = self._get_types(_to_id(pkmn.get("species", "")))
                if types:
                    eff = self._type_eff_against(atk_type, types)
                    if eff >= 2.0:
                        se_count += 1
                    if eff < 1.0:
                        resisted = True
                    if eff <= 0.25:
                        doubly_resisted += 1
            worst = max(worst, se_count)
            if not resisted:
                unresisted += 1

        # Core detection (FWG or DSF)
        has_core = float(
            ({"Fire", "Water", "Grass"} <= unique_types)
            or ({"Dragon", "Steel", "Fairy"} <= unique_types)
        )

        return [
            unique_count / NUM_TYPES,
            redundancy,
            worst / 6.0,
            unresisted / NUM_TYPES,
            doubly_resisted / (NUM_TYPES * 6.0),
            has_core,
        ]

    # ------------------------------------------------------------------
    # New XGBoost feature groups (Phase 2)
    # ------------------------------------------------------------------

    def _move_interaction_features(self, team: list[dict]) -> list[float]:
        """Score synergistic move/ability/item interactions within a team (10 features)."""
        setup_priority = 0
        setup_stab = 0
        hazard_phazing = 0
        pivot_hazard = 0
        choice_pivot = 0
        recovery_status = 0
        trick_room_slow = 0
        weather_synergy = 0
        terrain_synergy = 0
        regen_pivot = 0

        has_hazards = False
        has_phazing = False
        has_weather_setter = None
        has_terrain_setter = None
        slow_attackers = 0

        for pkmn in team:
            moves = {_to_id(m) for m in pkmn.get("moves", []) if m}
            item_id = _to_id(pkmn.get("item", "") or "")
            ability_id = _to_id(pkmn.get("ability", "") or "")
            species_id = _to_id(pkmn.get("species", ""))
            base = self._get_base_stats(species_id) or {}
            pkmn_types = self._get_types(species_id)

            has_setup = bool(moves & self.SETUP_MOVES)
            has_pivot = bool(moves & self.PIVOT_MOVES)
            has_prio = bool(moves & self.PRIORITY_MOVES)
            has_recov = bool(moves & self.RECOVERY_MOVES)
            has_status = bool(moves & self.STATUS_MOVES)

            # Setup + priority on same Mon
            if has_setup and has_prio:
                setup_priority += 1
            # Setup + STAB coverage
            if has_setup:
                stab_dmg = sum(1 for m in moves
                               if self._get_move(m) and
                               self._get_move(m).get("basePower", 0) > 0 and
                               self._get_move(m).get("type", "") in pkmn_types)
                if stab_dmg > 0:
                    setup_stab += 1
            # Choice + pivot
            if is_choice_item(item_id) and has_pivot:
                choice_pivot += 1
            # Recovery + status
            if has_recov and has_status:
                recovery_status += 1
            # Regen + pivot
            if ability_is_regenerator(ability_id) and has_pivot:
                regen_pivot += 1

            if moves & self.HAZARD_MOVES:
                has_hazards = True
            if moves & self.PHAZING_MOVES:
                has_phazing = True

            # Weather/terrain
            ab_eff = ABILITY_EFFECTS.get(ability_id, {})
            if ab_eff.get("weather"):
                has_weather_setter = ab_eff["weather"]
            if ab_eff.get("terrain"):
                has_terrain_setter = ab_eff["terrain"]

            # Slow for Trick Room
            if base.get("spe", 0) <= 50 and (base.get("atk", 0) >= 100 or base.get("spa", 0) >= 100):
                slow_attackers += 1

            # Trick Room check
            if "trickroom" in moves:
                trick_room_slow += slow_attackers

        # Cross-team synergies
        if has_hazards and has_phazing:
            hazard_phazing = 1
        for pkmn in team:
            moves = {_to_id(m) for m in pkmn.get("moves", []) if m}
            if has_hazards and (moves & self.PIVOT_MOVES):
                pivot_hazard += 1

        # Weather/terrain abuse
        if has_weather_setter:
            for pkmn in team:
                ab = ABILITY_EFFECTS.get(_to_id(pkmn.get("ability", "") or ""), {})
                if ab.get("weather_speed") == has_weather_setter:
                    weather_synergy += 1
        if has_terrain_setter:
            for pkmn in team:
                ab = ABILITY_EFFECTS.get(_to_id(pkmn.get("ability", "") or ""), {})
                if ab.get("terrain_or_booster") == has_terrain_setter:
                    terrain_synergy += 1

        return [
            setup_priority / 6.0, setup_stab / 6.0,
            float(hazard_phazing), pivot_hazard / 6.0,
            choice_pivot / 6.0, recovery_status / 6.0,
            min(trick_room_slow / 6.0, 1.0),
            weather_synergy / 6.0, terrain_synergy / 6.0,
            regen_pivot / 6.0,
        ]

    def _effective_power_features(self, team: list[dict]) -> list[float]:
        """Calculate effective power accounting for item/ability multipliers (8 features)."""
        eff_phys_powers = []
        eff_spec_powers = []
        wallbreaker_scores = []
        setup_sweeper_scores = []

        for pkmn in team:
            species_id = _to_id(pkmn.get("species", ""))
            base = unify_special_stat(self._get_base_stats(species_id) or {}, self.gen)
            pkmn_types = self._get_types(species_id)
            item_id = _to_id(pkmn.get("item", "") or "")
            ability_id = _to_id(pkmn.get("ability", "") or "")

            atk = base.get("atk", 80)
            spa = base.get("spa", 80)
            spe = base.get("spe", 80)

            phys_mult = get_item_atk_mult(item_id, True) * get_ability_atk_mult(ability_id)
            spec_mult = get_item_atk_mult(item_id, False) * get_ability_atk_mult(ability_id)
            dmg_mult = get_item_damage_mult(item_id)

            best_phys = 0
            best_spec = 0
            has_setup = False
            moves = set()
            for m in pkmn.get("moves", []):
                if not m:
                    continue
                mid = _to_id(m)
                moves.add(mid)
                md = self._get_move(m)
                if md:
                    bp = md.get("basePower", 0)
                    cat = get_move_category(md, self.gen)
                    mtype = md.get("type", "")
                    stab = 1.5 if mtype in pkmn_types else 1.0
                    if cat == "Physical" and bp > 0:
                        eff = bp * atk * phys_mult * dmg_mult * stab / 300.0
                        best_phys = max(best_phys, eff)
                    elif cat == "Special" and bp > 0:
                        eff = bp * spa * spec_mult * dmg_mult * stab / 300.0
                        best_spec = max(best_spec, eff)

            if moves & self.SETUP_MOVES:
                has_setup = True

            eff_phys_powers.append(best_phys)
            eff_spec_powers.append(best_spec)
            wallbreaker_scores.append(max(best_phys, best_spec))
            if has_setup:
                setup_sweeper_scores.append(max(best_phys, best_spec) * (spe / 200.0))

        mean_phys = float(np.mean(eff_phys_powers)) if eff_phys_powers else 0.0
        max_phys = max(eff_phys_powers) if eff_phys_powers else 0.0
        mean_spec = float(np.mean(eff_spec_powers)) if eff_spec_powers else 0.0
        max_spec = max(eff_spec_powers) if eff_spec_powers else 0.0
        best_wallbreaker = max(wallbreaker_scores) if wallbreaker_scores else 0.0
        best_sweeper = max(setup_sweeper_scores) if setup_sweeper_scores else 0.0
        mixed = sum(1 for p, s in zip(eff_phys_powers, eff_spec_powers)
                    if p > 30 and s > 30) / max(len(team), 1)
        choice_threat = sum(1 for pkmn in team
                           if is_choice_item(_to_id(pkmn.get("item", "") or "")))

        return [
            mean_phys / 100.0, max_phys / 100.0,
            mean_spec / 100.0, max_spec / 100.0,
            best_wallbreaker / 100.0, best_sweeper / 100.0,
            mixed, choice_threat / 6.0,
        ]

    def _enhanced_item_ability_numeric_features(self, team: list[dict]) -> list[float]:
        """Extract numeric item/ability multiplier features (8 features)."""
        total_atk_mult = 0.0
        total_spa_mult = 0.0
        total_def_mult = 0.0
        total_spd_mult = 0.0
        intimidate_count = 0
        regen_count = 0
        weather_count = 0
        terrain_count = 0

        for pkmn in team:
            item_id = _to_id(pkmn.get("item", "") or "")
            ability_id = _to_id(pkmn.get("ability", "") or "")

            total_atk_mult += get_item_atk_mult(item_id, True) * get_ability_atk_mult(ability_id)
            total_spa_mult += get_item_atk_mult(item_id, False)
            total_def_mult += get_item_def_mult(item_id, True)
            total_spd_mult += get_item_def_mult(item_id, False)

            if ability_is_intimidate(ability_id):
                intimidate_count += 1
            if ability_is_regenerator(ability_id):
                regen_count += 1
            if ability_sets_weather(ability_id):
                weather_count += 1
            if ability_sets_terrain(ability_id):
                terrain_count += 1

        n = max(len(team), 1)
        return [
            total_atk_mult / n / 2.0, total_spa_mult / n / 2.0,
            total_def_mult / n / 2.0, total_spd_mult / n / 2.0,
            intimidate_count / 6.0, regen_count / 6.0,
            weather_count / 6.0, terrain_count / 6.0,
        ]

    def _detailed_status_features(self, team: list[dict]) -> list[float]:
        """Extract detailed status features from move JSON (6 features)."""
        total_burn_chance = 0.0
        total_para_chance = 0.0
        total_sleep = 0.0
        toxic_layers = 0
        secondary_diversity = set()
        contact_count = 0

        for pkmn in team:
            for m in pkmn.get("moves", []):
                if not m:
                    continue
                mid = _to_id(m)
                md = self._get_move(m)
                if not md:
                    continue

                # Count contact moves for Rocky Helmet / Flame Body interaction
                if md.get("flags", {}).get("contact"):
                    contact_count += 1

                # Extract status chances from JSON
                status = md.get("status")
                secondary = md.get("secondary")
                secondaries = md.get("secondaries", [])

                if status == "brn" or mid in self.BURN_MOVES:
                    acc = md.get("accuracy", 100)
                    total_burn_chance += (acc if isinstance(acc, (int, float)) else 100) / 100.0
                elif secondary and secondary.get("status") == "brn":
                    total_burn_chance += secondary.get("chance", 0) / 100.0

                if status == "par" or mid in self.PARA_MOVES:
                    acc = md.get("accuracy", 100)
                    total_para_chance += (acc if isinstance(acc, (int, float)) else 100) / 100.0
                elif secondary and secondary.get("status") == "par":
                    total_para_chance += secondary.get("chance", 0) / 100.0

                if mid in self.SLEEP_MOVES:
                    acc = md.get("accuracy", 100)
                    total_sleep += (acc if isinstance(acc, (int, float)) else 75) / 100.0

                if mid == "toxicspikes":
                    toxic_layers += 1

                # Secondary effect diversity
                if secondary:
                    s = secondary.get("status") or secondary.get("volatileStatus") or ""
                    if s:
                        secondary_diversity.add(s)
                    if secondary.get("boosts"):
                        secondary_diversity.add("boosts")
                for sec in secondaries:
                    s = sec.get("status") or sec.get("volatileStatus") or ""
                    if s:
                        secondary_diversity.add(s)

        return [
            min(total_burn_chance, 6.0) / 6.0,
            min(total_para_chance, 6.0) / 6.0,
            min(total_sleep, 3.0) / 3.0,
            min(toxic_layers, 2) / 2.0,
            len(secondary_diversity) / 8.0,
            contact_count / 24.0,
        ]

    def _gimmick_features(self, team: list[dict]) -> list[float]:
        """Generation-specific gimmick features (8 features).

        [0] has_mega: team has a Mega Stone holder (Gen 6-7)
        [1] has_z_crystal: team has a Z-Crystal holder (Gen 7)
        [2] best_z_power: best Z-Move BP on team / 200 (Gen 7)
        [3] has_dynamax: format allows Dynamax (Gen 8)
        [4] best_max_power: best Max Move BP on team / 150 (Gen 8)
        [5] tera_type_diversity: unique Tera types / 6 (Gen 9)
        [6] tera_new_stab_count: team members gaining new STAB from Tera / 6 (Gen 9)
        [7] mega_plus_z: team has both Mega + Z-Crystal (Gen 7 legal)
        """
        from .mechanics import is_mega_stone, is_z_crystal, get_z_crystal_type, get_z_move_bp, get_max_move_bp

        feats = [0.0] * 8
        has_mega = False
        has_z = False
        best_z_bp = 0
        best_max_bp = 0
        tera_types = set()
        tera_new_stab = 0

        for pkmn in team:
            if not pkmn:
                continue
            item_id = _to_id(pkmn.get("item") or "")
            species_id = _to_id(pkmn.get("species", ""))

            # Mega (Gen 6-7)
            if self.gen in (6, 7) and is_mega_stone(item_id):
                has_mega = True

            # Z-Crystal (Gen 7)
            if self.gen == 7 and is_z_crystal(item_id):
                has_z = True
                z_type = get_z_crystal_type(item_id)
                if z_type:
                    for m in pkmn.get("moves", []):
                        if not m:
                            continue
                        md = self._get_move(m)
                        if md and md.get("category") in ("Physical", "Special"):
                            if md.get("type") == z_type:
                                bp = md.get("basePower", 0) or 0
                                if bp > 0:
                                    best_z_bp = max(best_z_bp, get_z_move_bp(bp))

            # Dynamax (Gen 8)
            if self.gen == 8:
                for m in pkmn.get("moves", []):
                    if not m:
                        continue
                    md = self._get_move(m)
                    if md and md.get("category") in ("Physical", "Special"):
                        bp = md.get("basePower", 0) or 0
                        if bp > 0:
                            best_max_bp = max(best_max_bp, get_max_move_bp(bp, md.get("type", "")))

            # Tera (Gen 9)
            if self.gen >= 9:
                tera = pkmn.get("tera_type")
                if tera:
                    tera_types.add(tera)
                    types = self._get_types(species_id)
                    if tera not in types:
                        tera_new_stab += 1

        feats[0] = 1.0 if has_mega else 0.0
        feats[1] = 1.0 if has_z else 0.0
        feats[2] = best_z_bp / 200.0
        feats[3] = 1.0 if self.gen == 8 else 0.0
        feats[4] = best_max_bp / 150.0
        feats[5] = len(tera_types) / 6.0
        feats[6] = tera_new_stab / 6.0
        feats[7] = 1.0 if (has_mega and has_z) else 0.0  # Gen 7 legal combo

        return feats

    # ------------------------------------------------------------------
    # Pre-digested team data for fast matchup evaluation
    # ------------------------------------------------------------------

    def precompute_team_data(self, team: list[dict]) -> list[dict]:
        """Pre-digest all Pokemon data for a team. Eliminates dict lookups
        during matchup evaluation (~10x faster matchup_features).

        Includes item/ability modifiers for the damage calculator.
        Call once per team, then pass the result to _matchup_features_fast().
        """
        data = []
        for pkmn in team:
            species_id = _to_id(pkmn.get("species", ""))
            types = self._get_types(species_id)
            stats = self._get_base_stats(species_id) or {}

            item_id = _to_id(pkmn.get("item", "") or "")
            ability_id = _to_id(pkmn.get("ability", "") or "")
            item_eff = ITEM_EFFECTS.get(item_id, {})
            ability_eff = ABILITY_EFFECTS.get(ability_id, {})

            # Pre-compute move data with type indices and full move dict for damage calc
            moves_data = []
            move_ids = set()
            stab_type_indices = {TYPE_TO_IDX[t] for t in types if t in TYPE_TO_IDX}
            for m in pkmn.get("moves", []):
                if not m:
                    continue
                mid = _to_id(m)
                move_ids.add(mid)
                md = self._get_move(m)
                if md:
                    mtype = md.get("type", "")
                    moves_data.append({
                        "type": mtype,
                        "type_idx": TYPE_TO_IDX.get(mtype, -1),
                        "basePower": md.get("basePower", 0),
                        "category": get_move_category(md, self.gen),
                        "is_stab": TYPE_TO_IDX.get(mtype, -1) in stab_type_indices,
                        # Full move data for damage_calc
                        "name": md.get("name", ""),
                        "flags": md.get("flags", {}),
                        "secondary": md.get("secondary"),
                        "secondaries": md.get("secondaries"),
                        "overrideOffensiveStat": md.get("overrideOffensiveStat"),
                        "overrideOffensivePokemon": md.get("overrideOffensivePokemon"),
                        "overrideDefensiveStat": md.get("overrideDefensiveStat"),
                    })

            # Gen 1: unify Special stat (SpA == SpD)
            stats = unify_special_stat(stats, self.gen)
            hp = stats.get("hp", 80)
            atk = stats.get("atk", 80)
            dfn = stats.get("def", 80)
            spa = stats.get("spa", 80)
            spd = stats.get("spd", 80)
            spe = stats.get("spe", 0)

            # Pre-compute defensive type effectiveness array
            def_eff = np.ones(NUM_TYPES, dtype=np.float32)
            for dt in types:
                dt_idx = TYPE_TO_IDX.get(dt)
                if dt_idx is None:
                    continue
                for atk_idx in range(NUM_TYPES):
                    atk_type = TYPES[atk_idx]
                    def_eff[atk_idx] *= self._type_eff(atk_type, dt)

            type_indices = [TYPE_TO_IDX[t] for t in types if t in TYPE_TO_IDX]

            # Item/ability multipliers for damage calculation
            atk_item_mult = get_item_atk_mult(item_id, True)
            spa_item_mult = get_item_atk_mult(item_id, False)
            def_item_mult = get_item_def_mult(item_id, True)
            spd_item_mult = get_item_def_mult(item_id, False)
            damage_mult = get_item_damage_mult(item_id)
            stab_mult = get_ability_stab_mult(ability_id)
            immune_type = get_ability_type_immunity(ability_id)

            data.append({
                "types": types,
                "type_indices": type_indices,
                "def_eff": def_eff,
                "spe": spe,
                "bst": hp + atk + dfn + spa + spd + spe,
                "hp": hp,
                "atk": atk,
                "def": dfn,
                "spa": spa,
                "spd": spd,
                "hp_actual": max(calc_stat(hp, self._stat_defaults["iv"],
                                          self._stat_defaults["ev"], LEVEL_100,
                                          1.0, True, self.gen), 1),
                "moves_data": moves_data,
                "move_ids": move_ids,
                "stab_types": set(types),
                "item_id": item_id,
                "ability_id": ability_id,
                "item_effects": item_eff,
                "ability_effects": ability_eff,
                "atk_item_mult": atk_item_mult,
                "spa_item_mult": spa_item_mult,
                "def_item_mult": def_item_mult,
                "spd_item_mult": spd_item_mult,
                "damage_mult": damage_mult,
                "stab_mult": stab_mult,
                "ability_immune_type": immune_type,
                "sr_dmg": self._sr_damage_fraction_from_types(types),
            })
        return data

    def _sr_damage_fraction_from_types(self, types: list[str]) -> float:
        """Stealth Rock damage from pre-computed types."""
        if not types:
            return 0.125
        eff = self._type_eff_against("Rock", types)
        return min(eff * 0.125, 0.5)

    def _matchup_features_fast(
        self, t1_data: list[dict], t2_data: list[dict]
    ) -> np.ndarray:
        """Compute matchup features from pre-digested team data.

        Same output as _matchup_features() but ~30x faster because all
        dict lookups and type effectiveness calls use pre-computed arrays.
        """
        features = []
        n1 = len(t1_data)
        n2 = len(t2_data)

        # Speed advantages
        speed_wins = 0
        speed_total = 0
        for p1 in t1_data:
            s1 = p1["spe"]
            for p2 in t2_data:
                s2 = p2["spe"]
                if s1 > 0 or s2 > 0:
                    speed_total += 1
                    if s1 > s2:
                        speed_wins += 1
        features.append(speed_wins / max(speed_total, 1))

        # SE count: team1 can hit team2 SE (using def_eff array)
        se_count = 0
        for p2 in t2_data:
            p2_def_eff = p2["def_eff"]
            if not p2["types"]:
                continue
            hit_se = False
            for p1 in t1_data:
                for md in p1["moves_data"]:
                    ti = md["type_idx"]
                    if ti >= 0 and md["basePower"] > 0 and md["category"] != "Status":
                        if p2_def_eff[ti] >= 2.0:
                            hit_se = True
                            break
                if hit_se:
                    break
            if hit_se:
                se_count += 1
        features.append(se_count / max(n2, 1))

        # Reverse SE: team2 hits team1 SE
        se_count_rev = 0
        for p1 in t1_data:
            p1_def_eff = p1["def_eff"]
            if not p1["types"]:
                continue
            hit_se = False
            for p2 in t2_data:
                for md in p2["moves_data"]:
                    ti = md["type_idx"]
                    if ti >= 0 and md["basePower"] > 0 and md["category"] != "Status":
                        if p1_def_eff[ti] >= 2.0:
                            hit_se = True
                            break
                if hit_se:
                    break
            if hit_se:
                se_count_rev += 1
        features.append(se_count_rev / max(n1, 1))

        # STAB resistance: team1 resists team2's STABs (using def_eff)
        resist_score = 0
        resist_total = 0
        for p2 in t2_data:
            for stab_idx in p2["type_indices"]:
                resist_total += 1
                for p1 in t1_data:
                    if p1["def_eff"][stab_idx] < 1.0:
                        resist_score += 1
                        break
        features.append(resist_score / max(resist_total, 1))

        # Reverse STAB resistance
        resist_score_rev = 0
        resist_total_rev = 0
        for p1 in t1_data:
            for stab_idx in p1["type_indices"]:
                resist_total_rev += 1
                for p2 in t2_data:
                    if p2["def_eff"][stab_idx] < 1.0:
                        resist_score_rev += 1
                        break
        features.append(resist_score_rev / max(resist_total_rev, 1))

        # BST advantage
        bst1 = sum(p["bst"] for p in t1_data) / max(n1, 1)
        bst2 = sum(p["bst"] for p in t2_data) / max(n2, 1)
        features.append((bst1 - bst2) / 720.0)

        # Threat matrix using full damage calculator with item/ability modifiers
        threat_t1 = []
        threat_t2 = []
        for p1 in t1_data:
            for p2 in t2_data:
                threat_t1.append(estimate_best_move_damage(p1, p2, type_chart=self._gen_chart, gen=self.gen))
        for p2 in t2_data:
            for p1 in t1_data:
                threat_t2.append(estimate_best_move_damage(p2, p1, type_chart=self._gen_chart, gen=self.gen))

        if threat_t1:
            features.extend([np.mean(threat_t1), max(threat_t1), min(threat_t1)])
        else:
            features.extend([0.0, 0.0, 0.0])
        if threat_t2:
            features.extend([np.mean(threat_t2), max(threat_t2), min(threat_t2)])
        else:
            features.extend([0.0, 0.0, 0.0])

        # 2HKO potential
        t1_2hko = sum(1 for t in threat_t1 if t > 0.45) / max(len(threat_t1), 1)
        t2_2hko = sum(1 for t in threat_t2 if t > 0.45) / max(len(threat_t2), 1)
        features.extend([t1_2hko, t2_2hko])

        # Safe switch-in count
        t1_safe = 0
        for j in range(n2):
            worst = max((threat_t1[i * n2 + j] for i in range(n1)), default=0)
            if worst < 0.25:
                t1_safe += 1
        t2_safe = 0
        for j in range(n1):
            worst = max((threat_t2[i * n1 + j] for i in range(n2)), default=0)
            if worst < 0.25:
                t2_safe += 1
        features.extend([t1_safe / max(n2, 1), t2_safe / max(n1, 1)])

        # SR damage
        sr_t1 = sum(p["sr_dmg"] for p in t1_data) / max(n1, 1)
        sr_t2 = sum(p["sr_dmg"] for p in t2_data) / max(n2, 1)
        features.extend([sr_t1, sr_t2])

        # STAB immunity count (using def_eff array: == 0 means immune)
        t1_immune = 0
        for p1 in t1_data:
            for stab_idx in p1["type_indices"]:
                for p2 in t2_data:
                    if p2["def_eff"][stab_idx] == 0:
                        t1_immune += 1
                        break
        t2_immune = 0
        for p2 in t2_data:
            for stab_idx in p2["type_indices"]:
                for p1 in t1_data:
                    if p1["def_eff"][stab_idx] == 0:
                        t2_immune += 1
                        break
        features.extend([t1_immune / max(n1 * 2, 1), t2_immune / max(n2 * 2, 1)])

        # Priority kill potential
        t1_prio = 0
        t2_prio = 0
        for p in t1_data:
            if p["move_ids"] & self.PRIORITY_MOVES and (p["atk"] >= 100 or p["spa"] >= 100):
                t1_prio += 1
        for p in t2_data:
            if p["move_ids"] & self.PRIORITY_MOVES and (p["atk"] >= 100 or p["spa"] >= 100):
                t2_prio += 1
        features.extend([t1_prio / 6.0, t2_prio / 6.0])

        # ---- Enhanced matchup features (8 new) ----

        # Ability immunity denial: t1 has moves of a type that t2 has ability immunity to (2)
        t1_denied = 0
        t2_denied = 0
        for p2 in t2_data:
            immune_type = p2.get("ability_immune_type")
            if immune_type:
                immune_idx = TYPE_TO_IDX.get(immune_type, -1)
                for p1 in t1_data:
                    for md in p1["moves_data"]:
                        if md["type_idx"] == immune_idx and md["basePower"] > 0:
                            t1_denied += 1
                            break
        for p1 in t1_data:
            immune_type = p1.get("ability_immune_type")
            if immune_type:
                immune_idx = TYPE_TO_IDX.get(immune_type, -1)
                for p2 in t2_data:
                    for md in p2["moves_data"]:
                        if md["type_idx"] == immune_idx and md["basePower"] > 0:
                            t2_denied += 1
                            break
        features.extend([t1_denied / max(n1 * n2, 1), t2_denied / max(n1 * n2, 1)])

        # Intimidate impact: fraction of opponent's physical attackers weakened (2)
        t1_intim = sum(1 for p in t1_data if ability_is_intimidate(p.get("ability_id", "")))
        t2_intim = sum(1 for p in t2_data if ability_is_intimidate(p.get("ability_id", "")))
        t2_phys = sum(1 for p in t2_data if p["atk"] > p["spa"])
        t1_phys = sum(1 for p in t1_data if p["atk"] > p["spa"])
        features.append(t1_intim * t2_phys / max(n2, 1) / 6.0)
        features.append(t2_intim * t1_phys / max(n1, 1) / 6.0)

        # Entry hazard vs non-Boots ratio (2)
        t1_boots = sum(1 for p in t1_data if is_hazard_immune_item(p.get("item_id", "")))
        t2_boots = sum(1 for p in t2_data if is_hazard_immune_item(p.get("item_id", "")))
        t1_has_hazards = any(p["move_ids"] & self.HAZARD_MOVES for p in t1_data)
        t2_has_hazards = any(p["move_ids"] & self.HAZARD_MOVES for p in t2_data)
        # If t1 has hazards, how vulnerable is t2 (non-boots fraction)?
        features.append((n2 - t2_boots) / max(n2, 1) if t1_has_hazards else 0.0)
        features.append((n1 - t1_boots) / max(n1, 1) if t2_has_hazards else 0.0)

        # Regenerator/pivot sustainability score (2)
        t1_regen_pivot = sum(
            1 for p in t1_data
            if ability_is_regenerator(p.get("ability_id", ""))
            and p["move_ids"] & self.PIVOT_MOVES
        )
        t2_regen_pivot = sum(
            1 for p in t2_data
            if ability_is_regenerator(p.get("ability_id", ""))
            and p["move_ids"] & self.PIVOT_MOVES
        )
        features.extend([t1_regen_pivot / 6.0, t2_regen_pivot / 6.0])

        return np.array(features, dtype=np.float32)

    def _estimate_threat_fast(self, atk_data: dict, def_data: dict) -> float:
        """Fast threat estimation from pre-digested Pokemon data.

        Uses item/ability modifiers from precompute_team_data().
        """
        return estimate_best_move_damage(atk_data, def_data, type_chart=self._gen_chart, gen=self.gen)

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
            if is_choice_item(item):
                flags["has_choice_item"] = 1

        return flags

    def get_feature_names(self) -> list[str]:
        """Return ordered list of XGBoost feature names for SHAP analysis."""
        team_names = (
            # Type composition (18)
            [f"type_{t.lower()}" for t in TYPES]
            # Offensive coverage (18)
            + [f"off_cov_{t.lower()}" for t in TYPES]
            # Defensive coverage (18)
            + [f"def_cov_{t.lower()}" for t in TYPES]
            # Base stat aggregates (24)
            + [f"base_{sn}_{agg}" for sn in STAT_NAMES for agg in ("mean", "std", "min", "max")]
            # Speed tier distribution (5)
            + [f"speed_tier_{i}" for i in range(5)]
            # BST aggregates (3)
            + ["bst_mean", "bst_min", "bst_max"]
            # Role indicators (6)
            + ["role_phys_atk", "role_spec_atk", "role_phys_wall",
               "role_spec_wall", "role_speed", "role_support"]
            # Utility flags (8)
            + ["has_hazards", "has_removal", "has_priority", "has_status",
               "has_recovery", "has_pivot", "has_setup", "has_choice"]
            # Move quality (20)
            + ["mq_total_bp", "mq_mean_bp", "mq_max_bp", "mq_std_bp",
               "mq_stab", "mq_priority", "mq_highpow", "mq_burn", "mq_para",
               "mq_sleep", "mq_toxic", "mq_secondary", "mq_pivot", "mq_setup",
               "mq_recovery", "mq_coverage", "mq_accuracy", "mq_spread",
               "mq_drain", "mq_recoil"]
            # Computed stat features (12)
            + ["cs_mean_spe", "cs_max_spe", "cs_mean_hp", "cs_mean_atk",
               "cs_mean_spa", "cs_mean_def", "cs_mean_spd", "cs_fast_count",
               "cs_strong_count", "cs_phys_bulk", "cs_spec_bulk", "cs_eff_bulk"]
            # Item effect features (12)
            + ["item_choice", "item_lifeorb", "item_leftovers", "item_boots",
               "item_sash", "item_offensive", "item_defensive", "item_av",
               "item_eviolite", "item_berry", "item_terrain", "item_helmet"]
            # Ability effect features (12)
            + ["abi_immunity", "abi_weather", "abi_terrain", "abi_intimidate",
               "abi_speedboost", "abi_powerboost", "abi_hazardimmune",
               "abi_regenerator", "abi_statusimmune", "abi_prankster",
               "abi_contactpunish", "abi_moldbreaker"]
            # Type synergy features (6)
            + ["tsyn_unique", "tsyn_redundancy", "tsyn_worst_weak",
               "tsyn_unresisted", "tsyn_double_resist", "tsyn_has_core"]
            # Move interaction features (10)
            + ["mi_setup_priority", "mi_setup_stab", "mi_hazard_phazing",
               "mi_pivot_hazard", "mi_choice_pivot", "mi_recovery_status",
               "mi_trick_room", "mi_weather_synergy", "mi_terrain_synergy",
               "mi_regen_pivot"]
            # Effective power features (8)
            + ["ep_mean_phys", "ep_max_phys", "ep_mean_spec", "ep_max_spec",
               "ep_best_wallbreaker", "ep_best_sweeper", "ep_mixed_threat",
               "ep_choice_threat"]
            # Enhanced item/ability numeric features (8)
            + ["ean_team_atk_mult", "ean_team_spa_mult", "ean_team_def_mult",
               "ean_team_spd_mult", "ean_intimidate", "ean_regenerator",
               "ean_weather_setter", "ean_terrain_setter"]
            # Detailed status features (6)
            + ["ds_burn_chance", "ds_para_chance", "ds_sleep_induction",
               "ds_toxic_layers", "ds_secondary_diversity", "ds_contact_moves"]
        )

        names = []
        for prefix in ["t1", "t2", "diff"]:
            names.extend([f"{prefix}_{n}" for n in team_names])

        # Matchup features (30 = 22 original + 8 new)
        names.extend([
            "mu_speed_adv", "mu_se_t1", "mu_se_t2",
            "mu_stab_resist_t1", "mu_stab_resist_t2", "mu_bst_adv",
            "mu_threat_t1_mean", "mu_threat_t1_max", "mu_threat_t1_min",
            "mu_threat_t2_mean", "mu_threat_t2_max", "mu_threat_t2_min",
            "mu_2hko_t1", "mu_2hko_t2",
            "mu_safe_switch_t1", "mu_safe_switch_t2",
            "mu_sr_dmg_t1", "mu_sr_dmg_t2",
            "mu_stab_immune_t1", "mu_stab_immune_t2",
            "mu_prio_threat_t1", "mu_prio_threat_t2",
            # New enhanced matchup features
            "mu_ability_denied_t1", "mu_ability_denied_t2",
            "mu_intimidate_impact_t1", "mu_intimidate_impact_t2",
            "mu_hazard_vuln_t2", "mu_hazard_vuln_t1",
            "mu_regen_pivot_t1", "mu_regen_pivot_t2",
        ])

        # Rating features (6)
        names.extend([
            "rating_normalized", "rating_has_data", "rating_above_avg",
            "rating_high", "rating_above_offset", "rating_low_flag",
        ])

        return names
