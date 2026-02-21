"""Advanced team analysis: speed tiers, archetype detection, counter-teaming, tera optimization."""

import logging
import re
from typing import Any

import numpy as np

from ..utils.constants import (
    TYPES, TYPE_TO_IDX, NUM_TYPES, STAT_NAMES, NATURES,
    type_effectiveness_against, calc_stat,
    IV_DEFAULT, EV_MAX_SINGLE, LEVEL_100,
)

log = logging.getLogger("showdown.teambuilder.analysis")


def _to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class TeamAnalyzer:
    """Advanced analysis tools for Pokemon teams."""

    def __init__(self, pokemon_data=None):
        self.pokemon_data = pokemon_data

    # ------------------------------------------------------------------
    # Speed Tier Analysis
    # ------------------------------------------------------------------

    # Key speed benchmarks at level 100 (base * nature * investment)
    SPEED_TIERS = [
        (648, "Max Regieleki (252+ Spe)"),
        (550, "Scarf base 130+ mons"),
        (502, "Max+ base 150"),
        (460, "Scarf base 110"),
        (417, "Max+ base 130"),
        (394, "Max base 130 / Scarf base 95"),
        (372, "Max+ base 115"),
        (350, "Max+ base 108"),
        (339, "Max+ base 102 / Max base 115"),
        (329, "Max base 110"),
        (309, "Max+ base 95 / Max base 102"),
        (299, "Max base 97"),
        (289, "Max base 95"),
        (278, "Max+ base 85"),
        (265, "Max base 85"),
        (244, "Max+ base 72"),
        (229, "Max base 72"),
        (207, "Max base 60"),
        (178, "Max base 50"),
        (139, "Max base 30"),
    ]

    def speed_tier_analysis(self, team: list[dict]) -> list[dict]:
        """Analyze each Pokemon's speed tier relative to the metagame.

        Returns list of speed tier info per Pokemon.
        """
        results = []
        for pkmn in team:
            species_id = _to_id(pkmn.get("species", ""))
            stats = self._get_base_stats(species_id)
            if not stats:
                results.append({
                    "species": pkmn.get("species", ""),
                    "base_speed": 0,
                    "min_speed": 0,
                    "max_speed": 0,
                    "tier": "Unknown",
                    "outspeeds": [],
                    "outspeed_by": [],
                })
                continue

            base_spe = stats.get("spe", 0)

            # Calculate speed range
            nature = pkmn.get("nature", "")
            evs = pkmn.get("evs", {})
            spe_ev = evs.get("spe", 0) if evs else 0

            # Max speed: 252 EV, 31 IV, +Speed nature
            max_speed = calc_stat(base_spe, IV_DEFAULT, EV_MAX_SINGLE, LEVEL_100, 1.1, False)
            # Neutral max: 252 EV, 31 IV, neutral nature
            neutral_speed = calc_stat(base_spe, IV_DEFAULT, EV_MAX_SINGLE, LEVEL_100, 1.0, False)
            # Min speed (for Trick Room): 0 EV, 0 IV, -Speed nature
            min_speed = calc_stat(base_spe, 0, 0, LEVEL_100, 0.9, False)

            # Actual speed if EVs known
            actual_nature_mult = 1.0
            if nature:
                ndata = NATURES.get(nature, {})
                actual_nature_mult = ndata.get("spe", 1.0)
            actual_speed = calc_stat(base_spe, IV_DEFAULT, spe_ev, LEVEL_100, actual_nature_mult, False)

            # Find speed tier
            tier_name = "Below all benchmarks"
            for threshold, name in self.SPEED_TIERS:
                if max_speed >= threshold:
                    tier_name = name
                    break

            # What does this outspeed / get outsped by
            outspeeds = []
            outspeed_by = []
            for threshold, name in self.SPEED_TIERS:
                if max_speed >= threshold:
                    outspeeds.append(name)
                elif max_speed < threshold:
                    outspeed_by.append(name)

            # Check for Scarf/Booster Energy speed
            item = _to_id(pkmn.get("item", "") or "")
            scarf_speed = None
            if item == "choicescarf":
                scarf_speed = int(actual_speed * 1.5)

            result = {
                "species": pkmn.get("species", ""),
                "base_speed": int(base_spe),
                "max_speed": int(max_speed),
                "neutral_speed": int(neutral_speed),
                "min_speed": int(min_speed),
                "actual_speed": int(actual_speed),
                "tier": tier_name,
                "outspeeds": outspeeds[:5],
                "outspeed_by": outspeed_by[:3],
            }
            if scarf_speed:
                result["scarf_speed"] = int(scarf_speed)

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Archetype Detection
    # ------------------------------------------------------------------

    def detect_archetype(self, team: list[dict]) -> dict[str, Any]:
        """Classify a team's playstyle archetype.

        Categories:
        - Hyper Offense (HO): Fast + setup heavy, minimal bulk
        - Bulky Offense (BO): Mix of power and bulk
        - Balance: Well-rounded, hazards + removal + mixed roles
        - Stall: Very defensive, passive damage, healing
        - Weather: Built around a weather condition
        - Trick Room: Slow team with TR setter
        """
        # Gather team stats
        speeds = []
        bulks = []  # (hp * def + hp * spd) / 2 as rough bulk metric
        setup_count = 0
        hazard_count = 0
        removal_count = 0
        recovery_count = 0
        pivot_count = 0
        weather_setters = []
        trick_room = False
        total_atk_power = 0

        SETUP_MOVES = {
            "swordsdance", "nastyplot", "dragondance", "calmmind", "bulkup",
            "quiverdance", "shellsmash", "agility", "bellydrum", "coil",
            "irondefense", "shiftgear", "victorydance", "tidyup",
        }
        HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
        REMOVAL_MOVES = {"rapidspin", "defog", "courtchange", "tidyup"}
        RECOVERY_MOVES = {
            "recover", "softboiled", "roost", "slackoff", "wish",
            "moonlight", "morningsun", "synthesis", "shoreup", "rest", "strengthsap",
        }
        PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "shedtail"}
        WEATHER_ABILITIES = {
            "drought": "sun", "drizzle": "rain", "sandstream": "sand",
            "snowwarning": "snow", "desolateland": "sun", "primordialsea": "rain",
        }
        TR_MOVES = {"trickroom"}

        for pkmn in team:
            species_id = _to_id(pkmn.get("species", ""))
            stats = self._get_base_stats(species_id)
            if stats:
                speeds.append(stats.get("spe", 0))
                hp = stats.get("hp", 0)
                dfn = stats.get("def", 0)
                spd = stats.get("spd", 0)
                bulks.append((hp * dfn + hp * spd) / 2)
                total_atk_power += max(stats.get("atk", 0), stats.get("spa", 0))

            moves = {_to_id(m) for m in pkmn.get("moves", []) if m}
            ability = _to_id(pkmn.get("ability", "") or "")

            if moves & SETUP_MOVES:
                setup_count += 1
            if moves & HAZARD_MOVES:
                hazard_count += 1
            if moves & REMOVAL_MOVES:
                removal_count += 1
            if moves & RECOVERY_MOVES:
                recovery_count += 1
            if moves & PIVOT_MOVES:
                pivot_count += 1
            if moves & TR_MOVES:
                trick_room = True

            if ability in WEATHER_ABILITIES:
                weather_setters.append(WEATHER_ABILITIES[ability])

        avg_speed = np.mean(speeds) if speeds else 0
        avg_bulk = np.mean(bulks) if bulks else 0
        avg_atk = total_atk_power / max(len(team), 1)

        # Classify
        scores = {}
        scores["Hyper Offense"] = (
            (setup_count >= 3) * 3 +
            (avg_speed >= 95) * 2 +
            (recovery_count <= 1) * 1 +
            (hazard_count >= 1) * 1
        )
        scores["Bulky Offense"] = (
            (setup_count >= 1) * 1 +
            (avg_bulk >= 25000) * 2 +
            (avg_atk >= 100) * 2 +
            (pivot_count >= 1) * 1 +
            (hazard_count >= 1) * 1
        )
        scores["Balance"] = (
            (hazard_count >= 1) * 2 +
            (removal_count >= 1) * 2 +
            (pivot_count >= 1) * 1 +
            (recovery_count >= 2) * 1 +
            (1 <= setup_count <= 2) * 1
        )
        scores["Stall"] = (
            (recovery_count >= 4) * 3 +
            (hazard_count >= 1) * 1 +
            (avg_bulk >= 35000) * 2 +
            (setup_count == 0) * 1
        )
        if weather_setters:
            weather = weather_setters[0]
            scores[f"Weather ({weather.title()})"] = 5
        if trick_room and avg_speed < 70:
            scores["Trick Room"] = 6

        archetype = max(scores, key=scores.get)

        return {
            "archetype": archetype,
            "scores": {k: int(v) for k, v in scores.items()},
            "stats": {
                "avg_speed": float(round(avg_speed, 1)),
                "avg_bulk": float(round(avg_bulk, 1)),
                "avg_attack_power": float(round(avg_atk, 1)),
                "setup_count": int(setup_count),
                "hazards": int(hazard_count),
                "removal": int(removal_count),
                "recovery": int(recovery_count),
                "pivots": int(pivot_count),
            },
        }

    # ------------------------------------------------------------------
    # Tera Type Optimization
    # ------------------------------------------------------------------

    def optimize_tera_types(self, team: list[dict], meta_teams: list[list[dict]]) -> list[dict]:
        """Suggest optimal tera types for each team member.

        Considers:
        - Removing key weaknesses
        - Gaining STAB on coverage moves
        - Defensive utility (Steel/Fairy/Ghost tera for resists)
        """
        suggestions = []

        for pkmn in team:
            species_id = _to_id(pkmn.get("species", ""))
            types = self._get_types(species_id)
            stats = self._get_base_stats(species_id)
            moves = pkmn.get("moves", [])

            if not types:
                suggestions.append({
                    "species": pkmn.get("species", ""),
                    "current_tera": pkmn.get("tera_type", ""),
                    "suggestions": [],
                })
                continue

            tera_scores = {}

            for tera_type in TYPES:
                score = 0.0

                # 1. Defensive benefit: does tera remove a weakness?
                for atk_type in TYPES:
                    orig_eff = type_effectiveness_against(atk_type, types)
                    tera_eff = type_effectiveness_against(atk_type, [tera_type])
                    if orig_eff >= 2.0 and tera_eff <= 1.0:
                        score += 2.0  # Remove a weakness
                    elif orig_eff >= 2.0 and tera_eff < orig_eff:
                        score += 1.0  # Reduce a weakness

                # 2. Offensive benefit: STAB bonus on moves
                for move_name in moves:
                    move_data = self._get_move(move_name)
                    if move_data and move_data.get("category") != "Status":
                        move_type = move_data.get("type", "")
                        if move_type == tera_type and tera_type not in types:
                            # Gain new STAB
                            score += 1.5

                # 3. High-value defensive tera types
                if tera_type == "Steel":
                    score += 0.5  # Many resistances
                elif tera_type == "Fairy":
                    score += 0.3  # Dragon immunity
                elif tera_type == "Ghost":
                    score += 0.3  # Normal/Fighting immunity

                # 4. Same-type tera (2x STAB boost)
                if tera_type in types:
                    score += 0.5  # Stronger existing STAB

                tera_scores[tera_type] = score

            # Sort and take top suggestions
            ranked = sorted(tera_scores.items(), key=lambda x: -x[1])
            top_suggestions = [
                {"type": t, "score": round(s, 1), "reason": self._tera_reason(t, types, moves)}
                for t, s in ranked[:3]
                if s > 0
            ]

            suggestions.append({
                "species": pkmn.get("species", ""),
                "current_tera": pkmn.get("tera_type", ""),
                "suggestions": top_suggestions,
            })

        return suggestions

    def _tera_reason(self, tera_type: str, orig_types: list[str], moves: list[str]) -> str:
        """Generate human-readable reason for a tera suggestion."""
        reasons = []

        # Check weakness removal
        for atk_type in TYPES:
            orig_eff = type_effectiveness_against(atk_type, orig_types)
            tera_eff = type_effectiveness_against(atk_type, [tera_type])
            if orig_eff >= 2.0 and tera_eff <= 0.5:
                reasons.append(f"Removes {atk_type} weakness")

        # Check STAB gain
        for move_name in moves:
            move_data = self._get_move(move_name)
            if move_data and move_data.get("type") == tera_type and tera_type not in orig_types:
                reasons.append(f"Gains STAB on {move_data.get('name', move_name)}")

        if tera_type in orig_types:
            reasons.append("Boosts existing STAB")

        return "; ".join(reasons[:2]) if reasons else "General defensive utility"

    # ------------------------------------------------------------------
    # Threat Analysis (Counter-teaming)
    # ------------------------------------------------------------------

    def analyze_threats(
        self, team: list[dict], meta_teams: list[list[dict]]
    ) -> dict[str, Any]:
        """Identify the biggest threats to this team from the meta.

        Returns species that most frequently have super-effective coverage
        against the team with few resistors.
        """
        if not self.pokemon_data:
            return {"threats": []}

        threat_scores = {}

        for meta_team in meta_teams:
            for threat in meta_team:
                threat_sp = threat.get("species", "")
                threat_id = _to_id(threat_sp)
                threat_types = self._get_types(threat_id)
                if not threat_types:
                    continue

                score = 0
                for pkmn in team:
                    sp_id = _to_id(pkmn.get("species", ""))
                    def_types = self._get_types(sp_id)
                    if not def_types:
                        continue

                    # Check if threat can hit super-effectively
                    for move_name in threat.get("moves", []):
                        move_data = self._get_move(move_name)
                        if move_data and move_data.get("category") != "Status":
                            eff = type_effectiveness_against(
                                move_data.get("type", ""), def_types
                            )
                            if eff >= 2.0:
                                score += 1
                                break

                    # Check if team member can threaten back
                    can_threaten = False
                    for move_name in pkmn.get("moves", []):
                        move_data = self._get_move(move_name)
                        if move_data and move_data.get("category") != "Status":
                            eff = type_effectiveness_against(
                                move_data.get("type", ""), threat_types
                            )
                            if eff >= 2.0:
                                can_threaten = True
                                break
                    if not can_threaten:
                        score += 0.5  # Extra threat if team can't hit back SE

                if threat_sp not in threat_scores:
                    threat_scores[threat_sp] = 0
                threat_scores[threat_sp] += score

        ranked = sorted(threat_scores.items(), key=lambda x: -x[1])[:10]

        return {
            "threats": [
                {
                    "species": sp,
                    "threat_score": round(score, 1),
                    "sprite": f"https://play.pokemonshowdown.com/sprites/gen5/{_to_id(sp)}.png",
                }
                for sp, score in ranked
            ],
        }

    # ------------------------------------------------------------------
    # Coverage Analysis
    # ------------------------------------------------------------------

    def coverage_analysis(self, team: list[dict]) -> dict[str, Any]:
        """Analyze the team's offensive and defensive type coverage."""
        # Offensive: which types can we hit SE?
        offensive = {}
        for def_type in TYPES:
            can_hit = []
            for pkmn in team:
                for move_name in pkmn.get("moves", []):
                    move_data = self._get_move(move_name)
                    if move_data and move_data.get("category") != "Status":
                        eff = type_effectiveness_against(move_data.get("type", ""), [def_type])
                        if eff >= 2.0:
                            can_hit.append(pkmn.get("species", ""))
                            break
            offensive[def_type] = can_hit

        # Defensive: which types are we weak to?
        defensive = {}
        for atk_type in TYPES:
            weak_to = []
            resists = []
            for pkmn in team:
                sp_id = _to_id(pkmn.get("species", ""))
                def_types = self._get_types(sp_id)
                if not def_types:
                    continue
                eff = type_effectiveness_against(atk_type, def_types)
                if eff >= 2.0:
                    weak_to.append(pkmn.get("species", ""))
                elif eff < 1.0:
                    resists.append(pkmn.get("species", ""))
            defensive[atk_type] = {"weak": weak_to, "resist": resists}

        # Find uncovered types (can't hit SE)
        uncovered_offense = [t for t, hitters in offensive.items() if not hitters]
        # Find unresisted types (no resistor on team)
        unresisted_defense = [t for t, d in defensive.items() if not d["resist"]]

        return {
            "offensive_coverage": {t: int(len(h)) for t, h in offensive.items()},
            "uncovered_types": uncovered_offense,
            "defensive_weaknesses": {t: int(len(d["weak"])) for t, d in defensive.items() if d["weak"]},
            "unresisted_types": unresisted_defense,
            "offensive_detail": {t: list(h) for t, h in offensive.items() if h},
            "defensive_detail": {t: {"weak": list(d["weak"]), "resist": list(d["resist"])} for t, d in defensive.items() if d["weak"] or d["resist"]},
        }

    # ------------------------------------------------------------------
    # Helpers
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
