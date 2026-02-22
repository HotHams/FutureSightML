"""Turn-by-turn Pokemon battle simulator with Monte Carlo win-rate estimation.

Implements core Gen 9 singles mechanics:
- Damage formula with type effectiveness, STAB, critical hits
- Speed-based turn order with priority moves
- Status conditions (burn, poison, toxic, paralysis, sleep, freeze)
- Entry hazards (Stealth Rock, Spikes, Toxic Spikes)
- Key items (Choice Band/Specs/Scarf, Leftovers, Life Orb, Focus Sash, Boots)
- Key abilities (Intimidate, Levitate, Multiscale, Sturdy, Flash Fire, etc.)
- Weather (Sun, Rain, Sand, Snow) and terrain basics
- Terastallization
"""

from __future__ import annotations

import logging
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..data.pokemon_data import PokemonDataLoader, _to_id
from ..utils.constants import (
    NATURES,
    PHYSICAL,
    SPECIAL,
    STATUS,
    STAT_NAMES,
    calc_stat,
    type_effectiveness_against,
)

log = logging.getLogger("showdown.simulator")

# ---------------------------------------------------------------------------
# Enums & helpers
# ---------------------------------------------------------------------------

class StatusCondition(Enum):
    NONE = ""
    BURN = "brn"
    POISON = "psn"
    TOXIC = "tox"
    PARALYSIS = "par"
    SLEEP = "slp"
    FREEZE = "frz"

class Weather(Enum):
    NONE = ""
    SUN = "sun"
    RAIN = "rain"
    SAND = "sand"
    SNOW = "snow"

class HazardState:
    __slots__ = ("stealth_rock", "spikes", "toxic_spikes", "sticky_web")
    def __init__(self):
        self.stealth_rock = False
        self.spikes = 0       # 0-3 layers
        self.toxic_spikes = 0 # 0-2 layers
        self.sticky_web = False

# Commonly boosted item multipliers
ITEM_BOOSTS = {
    "choiceband": ("atk", 1.5),
    "choicespecs": ("spa", 1.5),
    "lifeorb": ("damage", 1.3),
}

# Abilities that grant immunities
IMMUNITY_ABILITIES = {
    "levitate": "Ground",
    "flashfire": "Fire",
    "waterabsorb": "Water",
    "voltabsorb": "Electric",
    "lightningrod": "Electric",
    "stormdrain": "Water",
    "dryskin": "Water",
    "motordrive": "Electric",
    "sapsipper": "Grass",
    "eartheater": "Ground",
}

# ---------------------------------------------------------------------------
# Battle Pokemon
# ---------------------------------------------------------------------------

@dataclass
class BattlePokemon:
    """A single Pokemon on the battlefield with computed stats and battle state."""
    species: str
    types: list[str]
    base_stats: dict[str, int]
    moves: list[str]
    ability: str
    item: str
    tera_type: str
    level: int
    nature: str
    evs: dict[str, int]

    # Computed at init
    stats: dict[str, int] = field(default_factory=dict)
    max_hp: int = 0
    current_hp: int = 0

    # Battle state
    status: StatusCondition = StatusCondition.NONE
    toxic_counter: int = 0
    sleep_turns: int = 0
    stat_stages: dict[str, int] = field(default_factory=dict)
    is_terastallized: bool = False
    has_moved: bool = False  # Track for Focus Sash
    substitute_hp: int = 0
    is_fainted: bool = False

    def __post_init__(self):
        if not self.stats:
            self._compute_stats()
        if self.max_hp == 0:
            self.max_hp = self.stats.get("hp", 1)
            self.current_hp = self.max_hp
        if not self.stat_stages:
            self.stat_stages = {s: 0 for s in STAT_NAMES if s != "hp"}

    def _compute_stats(self):
        nature_mults = NATURES.get(self.nature, {})
        for stat in STAT_NAMES:
            base = self.base_stats.get(stat, 80)
            ev = self.evs.get(stat, 0)
            nm = nature_mults.get(stat, 1.0)
            self.stats[stat] = calc_stat(base, 31, ev, self.level, nm, stat == "hp")

    @property
    def alive(self) -> bool:
        return self.current_hp > 0 and not self.is_fainted

    @property
    def hp_fraction(self) -> float:
        return self.current_hp / self.max_hp if self.max_hp > 0 else 0.0

    def effective_stat(self, stat: str) -> int:
        """Get the effective stat with stage modifiers and status."""
        base = self.stats.get(stat, 1)
        stage = self.stat_stages.get(stat, 0)
        if stage >= 0:
            mult = (2 + stage) / 2
        else:
            mult = 2 / (2 - stage)
        val = int(base * mult)
        # Burn halves physical attack
        if stat == "atk" and self.status == StatusCondition.BURN:
            val = val // 2
        # Paralysis halves speed
        if stat == "spe" and self.status == StatusCondition.PARALYSIS:
            val = val // 2
        return max(1, val)

    def effective_types(self) -> list[str]:
        """Current types, considering terastallization."""
        if self.is_terastallized and self.tera_type:
            return [self.tera_type]
        return self.types

    def take_damage(self, amount: int) -> int:
        """Apply damage, return actual damage dealt."""
        amount = max(0, amount)
        actual = min(self.current_hp, amount)
        self.current_hp -= actual
        if self.current_hp <= 0:
            self.current_hp = 0
            self.is_fainted = True
        return actual

    def heal(self, amount: int) -> int:
        """Heal HP, return actual amount healed."""
        amount = max(0, amount)
        actual = min(self.max_hp - self.current_hp, amount)
        self.current_hp += actual
        return actual


# ---------------------------------------------------------------------------
# Battle Side
# ---------------------------------------------------------------------------

@dataclass
class BattleSide:
    """One player's side of the battle."""
    team: list[BattlePokemon]
    active_index: int = 0
    hazards: HazardState = field(default_factory=HazardState)
    tera_used: bool = False

    @property
    def active(self) -> BattlePokemon | None:
        if 0 <= self.active_index < len(self.team):
            pkmn = self.team[self.active_index]
            return pkmn if pkmn.alive else None
        return None

    @property
    def all_fainted(self) -> bool:
        return all(not p.alive for p in self.team)

    def next_alive(self) -> int | None:
        """Index of the next alive Pokemon, or None."""
        for i, p in enumerate(self.team):
            if p.alive and i != self.active_index:
                return i
        return None

    def switch_to(self, index: int) -> None:
        self.active_index = index


# ---------------------------------------------------------------------------
# Battle Simulator
# ---------------------------------------------------------------------------

class BattleSimulator:
    """Simulates a single Pokemon battle between two teams."""

    MAX_TURNS = 200

    def __init__(self, data: PokemonDataLoader, rng: random.Random | None = None):
        self.data = data
        self.rng = rng or random.Random()
        self._weather = Weather.NONE
        self._weather_turns = 0

    def build_pokemon(self, pset: dict[str, Any]) -> BattlePokemon:
        """Build a BattlePokemon from a set dict (as stored in DB or API)."""
        species_id = _to_id(pset.get("species", ""))
        pkmn_data = self.data.get_pokemon(species_id)

        types = pkmn_data.get("types", ["Normal"]) if pkmn_data else ["Normal"]
        base_stats = pkmn_data.get("baseStats", {s: 80 for s in STAT_NAMES}) if pkmn_data else {s: 80 for s in STAT_NAMES}

        # Infer reasonable EVs if not provided
        evs = pset.get("evs", {})
        if not evs:
            evs = _infer_evs(base_stats, pset.get("moves", []), self.data)

        # Infer nature if not provided
        nature = pset.get("nature", "")
        if not nature:
            nature = _infer_nature(base_stats, evs, pset.get("moves", []), self.data)

        moves = [_to_id(m) for m in pset.get("moves", []) if m][:4]
        ability = _to_id(pset.get("ability", "")) or ""
        item = _to_id(pset.get("item", "")) or ""
        tera_type = pset.get("tera_type", "") or ""

        return BattlePokemon(
            species=species_id,
            types=types,
            base_stats=base_stats,
            moves=moves,
            ability=ability,
            item=item,
            tera_type=tera_type,
            level=pset.get("level", 100),
            nature=nature,
            evs=evs,
        )

    def build_team(self, team: list[dict[str, Any]]) -> list[BattlePokemon]:
        return [self.build_pokemon(p) for p in team]

    def simulate(
        self,
        team1: list[dict[str, Any]],
        team2: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run a full battle simulation. Returns result dict."""
        side1 = BattleSide(team=self.build_team(team1))
        side2 = BattleSide(team=self.build_team(team2))

        self._weather = Weather.NONE
        self._weather_turns = 0

        # Apply lead hazard damage
        self._apply_switch_in(side1, side2)
        self._apply_switch_in(side2, side1)

        turn = 0
        log_lines: list[str] = []

        while turn < self.MAX_TURNS:
            turn += 1
            p1 = side1.active
            p2 = side2.active

            if p1 is None or p2 is None:
                break

            # Decide actions (simplified AI: pick best move)
            move1, should_switch1 = self._pick_action(side1, side2)
            move2, should_switch2 = self._pick_action(side2, side1)

            # Handle switches first
            if should_switch1:
                idx = side1.next_alive()
                if idx is not None:
                    side1.switch_to(idx)
                    self._apply_switch_in(side1, side2)
                    p1 = side1.active

            if should_switch2:
                idx = side2.next_alive()
                if idx is not None:
                    side2.switch_to(idx)
                    self._apply_switch_in(side2, side1)
                    p2 = side2.active

            if p1 is None or p2 is None:
                break

            # Determine turn order
            p1_first = self._goes_first(p1, move1, p2, move2)

            if p1_first:
                self._execute_turn(side1, move1, side2, log_lines, turn)
                if side2.active and side2.active.alive:
                    self._execute_turn(side2, move2, side1, log_lines, turn)
            else:
                self._execute_turn(side2, move2, side1, log_lines, turn)
                if side1.active and side1.active.alive:
                    self._execute_turn(side1, move1, side2, log_lines, turn)

            # End-of-turn effects
            self._end_of_turn(side1)
            self._end_of_turn(side2)

            # Handle faints and forced switches
            self._handle_faints(side1, side2)
            self._handle_faints(side2, side1)

            # Check win condition
            if side1.all_fainted or side2.all_fainted:
                break

        # Determine winner
        if side1.all_fainted and side2.all_fainted:
            winner = 0  # Tie
        elif side2.all_fainted:
            winner = 1
        elif side1.all_fainted:
            winner = 2
        else:
            # Timeout — compare remaining HP
            hp1 = sum(p.current_hp for p in side1.team if p.alive)
            hp2 = sum(p.current_hp for p in side2.team if p.alive)
            winner = 1 if hp1 >= hp2 else 2

        return {
            "winner": winner,
            "turns": turn,
            "team1_remaining": sum(1 for p in side1.team if p.alive),
            "team2_remaining": sum(1 for p in side2.team if p.alive),
            "team1_hp_pct": sum(p.hp_fraction for p in side1.team) / len(side1.team),
            "team2_hp_pct": sum(p.hp_fraction for p in side2.team) / len(side2.team),
        }

    # -----------------------------------------------------------------------
    # Turn execution
    # -----------------------------------------------------------------------

    def _execute_turn(
        self,
        attacker_side: BattleSide,
        move_id: str,
        defender_side: BattleSide,
        log_lines: list[str],
        turn: int,
    ) -> None:
        atk = attacker_side.active
        dfn = defender_side.active
        if not atk or not atk.alive or not dfn or not dfn.alive:
            return
        if not move_id:
            return

        move_data = self.data.get_move(move_id)
        if not move_data:
            return

        category = move_data.get("category", STATUS)

        # Accuracy check
        acc = move_data.get("accuracy")
        if isinstance(acc, (int, float)) and acc < 100:
            if self.rng.random() * 100 > acc:
                return  # Miss

        # Paralysis: 25% chance of full paralysis
        if atk.status == StatusCondition.PARALYSIS and self.rng.random() < 0.25:
            return

        # Sleep: decrement turns, can't move
        if atk.status == StatusCondition.SLEEP:
            atk.sleep_turns -= 1
            if atk.sleep_turns > 0:
                return
            atk.status = StatusCondition.NONE

        # Freeze: 20% chance to thaw each turn
        if atk.status == StatusCondition.FREEZE:
            if self.rng.random() < 0.20 or move_data.get("type") == "Fire":
                atk.status = StatusCondition.NONE
            else:
                return

        # Handle status moves
        if category == STATUS:
            self._apply_status_move(atk, dfn, move_data, attacker_side, defender_side)
            return

        # Calculate and apply damage
        damage = self._calc_damage(atk, dfn, move_data)
        if damage <= 0:
            return

        # Focus Sash
        if dfn.item == "focussash" and dfn.current_hp == dfn.max_hp and damage >= dfn.current_hp:
            damage = dfn.current_hp - 1
            dfn.item = ""  # Consumed

        # Sturdy ability
        if dfn.ability == "sturdy" and dfn.current_hp == dfn.max_hp and damage >= dfn.current_hp:
            damage = dfn.current_hp - 1

        dfn.take_damage(damage)

        # Life Orb recoil
        if atk.item == "lifeorb":
            atk.take_damage(max(1, atk.max_hp // 10))

        # Drain moves
        drain = move_data.get("drain")
        if drain and isinstance(drain, list) and len(drain) == 2:
            heal_amount = max(1, damage * drain[0] // drain[1])
            atk.heal(heal_amount)

        # Recoil moves
        recoil = move_data.get("recoil")
        if recoil and isinstance(recoil, list) and len(recoil) == 2:
            recoil_dmg = max(1, damage * recoil[0] // recoil[1])
            atk.take_damage(recoil_dmg)

        # Secondary effects
        secondary = move_data.get("secondary")
        if secondary and isinstance(secondary, dict):
            chance = secondary.get("chance", 100) / 100.0
            if self.rng.random() < chance:
                self._apply_secondary(atk, dfn, secondary)

        atk.has_moved = True

    def _calc_damage(
        self,
        attacker: BattlePokemon,
        defender: BattlePokemon,
        move_data: dict,
    ) -> int:
        """Gen 9 damage formula."""
        base_power = move_data.get("basePower", 0)
        if base_power == 0:
            return 0

        category = move_data.get("category", PHYSICAL)
        move_type = move_data.get("type", "Normal")

        # Attack and defense stats
        if category == PHYSICAL:
            atk_stat = attacker.effective_stat("atk")
            def_stat = defender.effective_stat("def")
        else:
            atk_stat = attacker.effective_stat("spa")
            def_stat = defender.effective_stat("spd")

        # Item boosts
        item_info = ITEM_BOOSTS.get(attacker.item)
        damage_mult = 1.0
        if item_info:
            boosted_stat, boost = item_info
            if boosted_stat == "atk" and category == PHYSICAL:
                atk_stat = int(atk_stat * boost)
            elif boosted_stat == "spa" and category == SPECIAL:
                atk_stat = int(atk_stat * boost)
            elif boosted_stat == "damage":
                damage_mult *= boost

        level = attacker.level

        # Base damage calc
        damage = ((2 * level // 5 + 2) * base_power * atk_stat // def_stat) // 50 + 2

        # Type effectiveness
        def_types = defender.effective_types()
        effectiveness = type_effectiveness_against(move_type, def_types)

        # Ability immunities
        if defender.ability in IMMUNITY_ABILITIES:
            immune_type = IMMUNITY_ABILITIES[defender.ability]
            if move_type == immune_type:
                effectiveness = 0.0

        if effectiveness == 0.0:
            return 0

        damage = int(damage * effectiveness)

        # STAB
        stab = 1.0
        atk_types = attacker.effective_types()
        if move_type in atk_types:
            stab = 1.5
        # Tera STAB: if terastallized and move matches tera type
        if attacker.is_terastallized and attacker.tera_type == move_type:
            if move_type in attacker.types:
                stab = 2.0  # Adaptability-like boost
            else:
                stab = 1.5
        damage = int(damage * stab)

        # Weather
        if self._weather == Weather.SUN:
            if move_type == "Fire":
                damage = int(damage * 1.5)
            elif move_type == "Water":
                damage = int(damage * 0.5)
        elif self._weather == Weather.RAIN:
            if move_type == "Water":
                damage = int(damage * 1.5)
            elif move_type == "Fire":
                damage = int(damage * 0.5)

        # Critical hit (1/24 chance, 1.5x)
        if self.rng.random() < 1 / 24:
            damage = int(damage * 1.5)

        # Random factor (85-100%)
        damage = int(damage * (self.rng.randint(85, 100) / 100))

        # Additional multipliers
        damage = int(damage * damage_mult)

        # Multiscale
        if defender.ability == "multiscale" and defender.current_hp == defender.max_hp:
            damage = damage // 2

        return max(1, damage)

    # -----------------------------------------------------------------------
    # Status moves
    # -----------------------------------------------------------------------

    def _apply_status_move(
        self,
        attacker: BattlePokemon,
        defender: BattlePokemon,
        move_data: dict,
        atk_side: BattleSide,
        def_side: BattleSide,
    ) -> None:
        move_id = _to_id(move_data.get("name", ""))

        # Hazard moves
        if move_id == "stealthrock":
            def_side.hazards.stealth_rock = True
            return
        if move_id == "spikes" and def_side.hazards.spikes < 3:
            def_side.hazards.spikes += 1
            return
        if move_id == "toxicspikes" and def_side.hazards.toxic_spikes < 2:
            def_side.hazards.toxic_spikes += 1
            return
        if move_id == "stickyweb":
            def_side.hazards.sticky_web = True
            return

        # Hazard removal
        if move_id in ("rapidspin", "defog"):
            if move_id == "defog":
                # Defog clears both sides
                def_side.hazards = HazardState()
                atk_side.hazards = HazardState()
            else:
                atk_side.hazards = HazardState()
            return

        # Stat boosting moves
        boosts = move_data.get("boosts")
        if boosts:
            target = move_data.get("target", "self")
            if target in ("self", "adjacentAllyOrSelf", "allySide", "allies"):
                self._apply_boosts(attacker, boosts)
            else:
                self._apply_boosts(defender, boosts)
            return

        # Self-boost from non-target moves (Swords Dance, Calm Mind, etc.)
        self_boost = move_data.get("selfBoost", {}).get("boosts")
        if self_boost:
            self._apply_boosts(attacker, self_boost)

        # Status-inflicting moves
        status_str = move_data.get("status")
        if status_str:
            self._inflict_status(defender, status_str)
            return

        # Recovery moves
        if move_data.get("flags", {}).get("heal"):
            attacker.heal(attacker.max_hp // 2)
            return

        # Handle specific healing moves by ID
        if move_id in ("recover", "roost", "softboiled", "slackoff", "milkdrink",
                        "moonlight", "morningsun", "synthesis", "shoreup"):
            attacker.heal(attacker.max_hp // 2)

    def _apply_boosts(self, target: BattlePokemon, boosts: dict) -> None:
        for stat, amount in boosts.items():
            if stat in target.stat_stages:
                old = target.stat_stages[stat]
                target.stat_stages[stat] = max(-6, min(6, old + amount))

    def _inflict_status(self, target: BattlePokemon, status_str: str) -> None:
        if target.status != StatusCondition.NONE:
            return  # Already statused

        types = target.effective_types()

        if status_str == "brn":
            if "Fire" in types:
                return
            target.status = StatusCondition.BURN
        elif status_str == "par":
            if "Electric" in types:
                return
            target.status = StatusCondition.PARALYSIS
        elif status_str in ("psn", "tox"):
            if "Poison" in types or "Steel" in types:
                return
            target.status = StatusCondition.TOXIC if status_str == "tox" else StatusCondition.POISON
            target.toxic_counter = 0
        elif status_str == "slp":
            target.status = StatusCondition.SLEEP
            target.sleep_turns = self.rng.randint(1, 3)
        elif status_str == "frz":
            if "Ice" in types:
                return
            target.status = StatusCondition.FREEZE

    def _apply_secondary(
        self, attacker: BattlePokemon, defender: BattlePokemon, secondary: dict
    ) -> None:
        """Apply secondary effect of a move."""
        status_str = secondary.get("status")
        if status_str:
            self._inflict_status(defender, status_str)

        boosts = secondary.get("boosts")
        if boosts:
            self._apply_boosts(defender, boosts)

        self_boost = secondary.get("self", {})
        if isinstance(self_boost, dict) and "boosts" in self_boost:
            self._apply_boosts(attacker, self_boost["boosts"])

    # -----------------------------------------------------------------------
    # Turn order
    # -----------------------------------------------------------------------

    def _goes_first(
        self,
        p1: BattlePokemon,
        move1: str,
        p2: BattlePokemon,
        move2: str,
    ) -> bool:
        """Determine if p1 goes first."""
        prio1 = self._get_priority(move1)
        prio2 = self._get_priority(move2)
        if prio1 != prio2:
            return prio1 > prio2

        spd1 = p1.effective_stat("spe")
        spd2 = p2.effective_stat("spe")
        # Choice Scarf
        if p1.item == "choicescarf":
            spd1 = int(spd1 * 1.5)
        if p2.item == "choicescarf":
            spd2 = int(spd2 * 1.5)

        if spd1 != spd2:
            return spd1 > spd2
        return self.rng.random() < 0.5  # Speed tie

    def _get_priority(self, move_id: str) -> int:
        if not move_id:
            return 0
        move_data = self.data.get_move(move_id)
        if move_data:
            return move_data.get("priority", 0)
        return 0

    # -----------------------------------------------------------------------
    # AI: action selection
    # -----------------------------------------------------------------------

    def _pick_action(
        self,
        my_side: BattleSide,
        opp_side: BattleSide,
    ) -> tuple[str, bool]:
        """Pick the best move or decide to switch. Returns (move_id, should_switch)."""
        me = my_side.active
        opp = opp_side.active
        if not me or not opp:
            return ("", False)

        # Score each move
        best_move = ""
        best_score = -999.0

        for move_id in me.moves:
            score = self._score_move(me, opp, move_id)
            if score > best_score:
                best_score = score
                best_move = move_id

        # Consider switching if we're at a big disadvantage
        switch_score = -999.0
        switch_idx = None
        for i, pkmn in enumerate(my_side.team):
            if not pkmn.alive or i == my_side.active_index:
                continue
            s = self._score_matchup(pkmn, opp)
            if s > switch_score:
                switch_score = s
                switch_idx = i

        # Switch if the bench Pokemon has a much better matchup
        if switch_idx is not None and switch_score > best_score + 30 and me.hp_fraction > 0.3:
            return ("", True)

        return (best_move, False)

    def _score_move(self, attacker: BattlePokemon, defender: BattlePokemon, move_id: str) -> float:
        """Heuristic score for how good a move is in the current situation."""
        move_data = self.data.get_move(move_id)
        if not move_data:
            return -100.0

        category = move_data.get("category", STATUS)
        move_type = move_data.get("type", "Normal")

        if category == STATUS:
            return self._score_status_move(attacker, defender, move_data)

        base_power = move_data.get("basePower", 0)
        if base_power == 0:
            return -50.0

        # Type effectiveness
        eff = type_effectiveness_against(move_type, defender.effective_types())
        if defender.ability in IMMUNITY_ABILITIES:
            if IMMUNITY_ABILITIES[defender.ability] == move_type:
                eff = 0.0
        if eff == 0:
            return -100.0

        # STAB
        stab = 1.5 if move_type in attacker.effective_types() else 1.0

        # Category advantage
        if category == PHYSICAL:
            offense = attacker.effective_stat("atk")
            defense = defender.effective_stat("def")
        else:
            offense = attacker.effective_stat("spa")
            defense = defender.effective_stat("spd")

        score = base_power * eff * stab * (offense / max(1, defense))

        # Priority bonus when opponent is low
        priority = move_data.get("priority", 0)
        if priority > 0 and defender.hp_fraction < 0.3:
            score *= 1.5

        # Accuracy penalty
        acc = move_data.get("accuracy")
        if isinstance(acc, (int, float)) and acc < 100:
            score *= acc / 100

        return score

    def _score_status_move(
        self, attacker: BattlePokemon, defender: BattlePokemon, move_data: dict
    ) -> float:
        """Score a status move."""
        move_id = _to_id(move_data.get("name", ""))
        score = 0.0

        # Setup moves are great early
        boosts = move_data.get("boosts", {})
        if boosts and move_data.get("target", "self") in ("self", "adjacentAllyOrSelf"):
            for stat, amt in boosts.items():
                if amt > 0 and attacker.stat_stages.get(stat, 0) < 4:
                    score += amt * 25
            if attacker.hp_fraction < 0.5:
                score *= 0.3  # Don't setup when low

        # Status infliction
        if move_data.get("status") and defender.status == StatusCondition.NONE:
            score += 30
            # Thunder Wave on fast threats
            if move_data["status"] == "par" and defender.effective_stat("spe") > attacker.effective_stat("spe"):
                score += 20

        # Hazards
        if move_id == "stealthrock":
            score += 35
        elif move_id in ("spikes", "toxicspikes"):
            score += 20

        # Recovery
        if move_id in ("recover", "roost", "softboiled", "slackoff"):
            if attacker.hp_fraction < 0.6:
                score += 40
            else:
                score += 5

        # Defog/Rapid Spin
        if move_id in ("defog", "rapidspin"):
            score += 15

        return score

    def _score_matchup(self, pkmn: BattlePokemon, opponent: BattlePokemon) -> float:
        """Score how good a Pokemon is against the opponent."""
        score = 0.0
        # Offensive pressure
        for move_id in pkmn.moves:
            move_data = self.data.get_move(move_id)
            if not move_data or move_data.get("category") == STATUS:
                continue
            eff = type_effectiveness_against(
                move_data.get("type", "Normal"), opponent.effective_types()
            )
            if eff >= 2.0:
                score += 30
            elif eff >= 1.0:
                score += 10

        # Defensive typing
        for move_id in opponent.moves:
            move_data = self.data.get_move(move_id)
            if not move_data or move_data.get("category") == STATUS:
                continue
            eff = type_effectiveness_against(
                move_data.get("type", "Normal"), pkmn.effective_types()
            )
            if eff >= 2.0:
                score -= 25
            elif eff <= 0.5:
                score += 15
            elif eff == 0:
                score += 30

        # Speed advantage
        if pkmn.effective_stat("spe") > opponent.effective_stat("spe"):
            score += 10

        # HP factor
        score *= pkmn.hp_fraction

        return score

    # -----------------------------------------------------------------------
    # Entry hazards & switch-in
    # -----------------------------------------------------------------------

    def _apply_switch_in(self, side: BattleSide, opp_side: BattleSide) -> None:
        """Apply entry hazard damage and effects on switch-in."""
        pkmn = side.active
        if not pkmn or not pkmn.alive:
            return

        # Heavy-Duty Boots immunity
        if pkmn.item == "heavydutyboots":
            return

        types = pkmn.effective_types()
        is_flying = "Flying" in types or pkmn.ability == "levitate"

        # Stealth Rock — type-based damage
        if side.hazards.stealth_rock:
            eff = type_effectiveness_against("Rock", types)
            sr_damage = max(1, int(pkmn.max_hp * eff / 8))
            pkmn.take_damage(sr_damage)

        # Spikes — grounded only
        if side.hazards.spikes > 0 and not is_flying:
            spike_pcts = {1: 1/8, 2: 1/6, 3: 1/4}
            pct = spike_pcts.get(side.hazards.spikes, 1/8)
            pkmn.take_damage(max(1, int(pkmn.max_hp * pct)))

        # Toxic Spikes — grounded only
        if side.hazards.toxic_spikes > 0 and not is_flying:
            if "Poison" in types:
                # Poison types absorb Toxic Spikes
                side.hazards.toxic_spikes = 0
            elif "Steel" not in types and pkmn.status == StatusCondition.NONE:
                if side.hazards.toxic_spikes >= 2:
                    pkmn.status = StatusCondition.TOXIC
                else:
                    pkmn.status = StatusCondition.POISON

        # Sticky Web — grounded only
        if side.hazards.sticky_web and not is_flying:
            pkmn.stat_stages["spe"] = max(-6, pkmn.stat_stages.get("spe", 0) - 1)

        # Intimidate
        opp = opp_side.active
        if opp and opp.alive and opp.ability == "intimidate":
            pkmn.stat_stages["atk"] = max(-6, pkmn.stat_stages.get("atk", 0) - 1)

    # -----------------------------------------------------------------------
    # End of turn
    # -----------------------------------------------------------------------

    def _end_of_turn(self, side: BattleSide) -> None:
        pkmn = side.active
        if not pkmn or not pkmn.alive:
            return

        # Weather damage
        if self._weather == Weather.SAND:
            if not any(t in pkmn.effective_types() for t in ("Rock", "Ground", "Steel")):
                if pkmn.ability not in ("sandforce", "sandveil", "sandrush", "overcoat", "magicguard"):
                    pkmn.take_damage(max(1, pkmn.max_hp // 16))

        # Status damage
        if pkmn.status == StatusCondition.BURN:
            pkmn.take_damage(max(1, pkmn.max_hp // 16))
        elif pkmn.status == StatusCondition.POISON:
            pkmn.take_damage(max(1, pkmn.max_hp // 8))
        elif pkmn.status == StatusCondition.TOXIC:
            pkmn.toxic_counter += 1
            pkmn.take_damage(max(1, pkmn.max_hp * pkmn.toxic_counter // 16))

        # Leftovers
        if pkmn.item == "leftovers":
            pkmn.heal(max(1, pkmn.max_hp // 16))

        # Black Sludge (poison types)
        if pkmn.item == "blacksludge":
            if "Poison" in pkmn.effective_types():
                pkmn.heal(max(1, pkmn.max_hp // 16))
            else:
                pkmn.take_damage(max(1, pkmn.max_hp // 16))

    # -----------------------------------------------------------------------
    # Faint handling
    # -----------------------------------------------------------------------

    def _handle_faints(self, side: BattleSide, opp_side: BattleSide) -> None:
        """Switch in the next Pokemon if the active one fainted."""
        pkmn = side.active
        if pkmn and not pkmn.alive:
            # Pick best matchup from bench
            best_idx = None
            best_score = -999.0
            opp = opp_side.active
            for i, p in enumerate(side.team):
                if not p.alive or i == side.active_index:
                    continue
                if opp and opp.alive:
                    score = self._score_matchup(p, opp)
                else:
                    score = p.hp_fraction * 100
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is not None:
                side.switch_to(best_idx)
                self._apply_switch_in(side, opp_side)


# ---------------------------------------------------------------------------
# Monte Carlo Simulator
# ---------------------------------------------------------------------------

class MonteCarloSimulator:
    """Run many battle simulations to estimate win probability."""

    def __init__(self, data: PokemonDataLoader, n_simulations: int = 100):
        self.data = data
        self.n_simulations = n_simulations

    def estimate_win_rate(
        self,
        team1: list[dict[str, Any]],
        team2: list[dict[str, Any]],
        n: int | None = None,
    ) -> dict[str, Any]:
        """Run n simulations and return aggregated statistics."""
        n = n or self.n_simulations
        results = {"team1_wins": 0, "team2_wins": 0, "ties": 0, "details": []}

        for i in range(n):
            sim = BattleSimulator(self.data, rng=random.Random(i))
            result = sim.simulate(team1, team2)
            results["details"].append(result)

            if result["winner"] == 1:
                results["team1_wins"] += 1
            elif result["winner"] == 2:
                results["team2_wins"] += 1
            else:
                results["ties"] += 1

        results["team1_win_rate"] = results["team1_wins"] / n
        results["team2_win_rate"] = results["team2_wins"] / n
        results["tie_rate"] = results["ties"] / n
        results["avg_turns"] = sum(d["turns"] for d in results["details"]) / n
        results["avg_team1_remaining"] = sum(d["team1_remaining"] for d in results["details"]) / n
        results["avg_team2_remaining"] = sum(d["team2_remaining"] for d in results["details"]) / n
        results["n_simulations"] = n

        return results


# ---------------------------------------------------------------------------
# Helpers for inferring missing set data
# ---------------------------------------------------------------------------

def _infer_evs(
    base_stats: dict[str, int],
    moves: list[str],
    data: PokemonDataLoader,
) -> dict[str, int]:
    """Infer a reasonable EV spread based on base stats and moves."""
    has_physical = False
    has_special = False
    for m in moves:
        md = data.get_move(_to_id(m)) if data else None
        if md:
            cat = md.get("category", "")
            if cat == PHYSICAL:
                has_physical = True
            elif cat == SPECIAL:
                has_special = True

    evs: dict[str, int] = {}
    # Speed investment for faster Pokemon
    if base_stats.get("spe", 0) >= 80:
        evs["spe"] = 252

    # Offensive investment
    if has_physical and not has_special:
        evs["atk"] = 252
    elif has_special and not has_physical:
        evs["spa"] = 252
    elif has_physical and has_special:
        # Mixed — split
        if base_stats.get("atk", 0) >= base_stats.get("spa", 0):
            evs["atk"] = 252
        else:
            evs["spa"] = 252
    else:
        # No damaging moves — probably a wall
        evs["hp"] = 252
        evs["def"] = 128
        evs["spd"] = 128
        return evs

    # Remaining in HP
    remaining = 510 - sum(evs.values())
    evs["hp"] = min(252, remaining)
    return evs


def _infer_nature(
    base_stats: dict[str, int],
    evs: dict[str, int],
    moves: list[str],
    data: PokemonDataLoader,
) -> str:
    """Infer a reasonable nature based on stats and moves."""
    has_physical = any(
        (data.get_move(_to_id(m)) or {}).get("category") == PHYSICAL
        for m in moves
    )
    has_special = any(
        (data.get_move(_to_id(m)) or {}).get("category") == SPECIAL
        for m in moves
    )

    if evs.get("spe", 0) >= 252:
        if has_physical and not has_special:
            return "Jolly"   # +Spe -SpA
        elif has_special:
            return "Timid"   # +Spe -Atk
    elif evs.get("atk", 0) >= 252:
        return "Adamant"     # +Atk -SpA
    elif evs.get("spa", 0) >= 252:
        return "Modest"      # +SpA -Atk
    elif evs.get("def", 0) >= 128:
        if has_physical:
            return "Impish"  # +Def -SpA
        else:
            return "Bold"    # +Def -Atk
    elif evs.get("spd", 0) >= 128:
        return "Calm"        # +SpD -Atk

    return "Hardy"  # Neutral
