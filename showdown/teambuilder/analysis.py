"""Advanced team analysis: speed tiers, archetype detection, counter-teaming, tera optimization."""

import logging
import re
from typing import Any

import numpy as np

from ..utils.constants import (
    TYPES, TYPE_TO_IDX, NUM_TYPES, STAT_NAMES, NATURES,
    type_effectiveness_against, calc_stat,
    IV_DEFAULT, EV_MAX_SINGLE, LEVEL_100,
    extract_gen,
)
from ..data.damage_calc import estimate_damage_pct, estimate_best_move_damage

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
    # Team Strategy Explainer
    # ------------------------------------------------------------------

    def explain_strategy(self, team: list[dict]) -> dict[str, Any]:
        """Analyze the team's intended strategy, roles, and win conditions.

        Returns a structured explanation of how the team is meant to function:
        - Each Pokemon's role and purpose
        - Primary and secondary win conditions
        - Defensive cores and synergies
        - Game plan (lead, mid-game, end-game)
        """
        roles = []
        win_conditions = []
        game_plan = {"lead_candidates": [], "mid_game": [], "end_game": []}
        synergies = []

        # Classify each Pokemon
        for pkmn in team:
            role = self._classify_role(pkmn)
            roles.append(role)

        # Identify win conditions
        win_conditions = self._identify_win_conditions(team, roles)

        # Identify defensive cores
        synergies = self._identify_synergies(team, roles)

        # Build game plan
        game_plan = self._build_game_plan(team, roles)

        # Generate summary
        summary = self._generate_summary(roles, win_conditions, synergies, game_plan)

        return {
            "roles": roles,
            "win_conditions": win_conditions,
            "synergies": synergies,
            "game_plan": game_plan,
            "summary": summary,
        }

    def _classify_role(self, pkmn: dict) -> dict:
        """Determine a Pokemon's role on the team."""
        species = pkmn.get("species", "")
        species_id = _to_id(species)
        stats = self._get_base_stats(species_id)
        types = self._get_types(species_id)
        moves = [m for m in pkmn.get("moves", []) if m]
        move_ids = {_to_id(m) for m in moves}
        ability = _to_id(pkmn.get("ability", "") or "")
        item = _to_id(pkmn.get("item", "") or "")

        role_tags = []
        description_parts = []

        if not stats:
            return {"species": species, "role": "Unknown", "tags": [], "description": ""}

        atk = stats.get("atk", 0)
        spa = stats.get("spa", 0)
        dfn = stats.get("def", 0)
        spd = stats.get("spd", 0)
        hp = stats.get("hp", 0)
        spe = stats.get("spe", 0)
        best_offense = max(atk, spa)
        phys_bulk = hp * dfn
        spec_bulk = hp * spd

        # Move category sets
        SETUP = {"swordsdance", "nastyplot", "dragondance", "calmmind",
                 "bulkup", "quiverdance", "shellsmash", "agility",
                 "bellydrum", "coil", "irondefense", "shiftgear",
                 "victorydance", "tidyup", "curse", "growth"}
        HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
        REMOVAL = {"rapidspin", "defog", "courtchange", "tidyup"}
        RECOVERY = {"recover", "softboiled", "roost", "slackoff", "wish",
                    "moonlight", "morningsun", "synthesis", "shoreup",
                    "strengthsap", "rest"}
        PIVOTS = {"uturn", "voltswitch", "flipturn", "partingshot",
                  "teleport", "shedtail", "batonpass"}
        STATUS_MOVES = {"willowisp", "thunderwave", "toxic", "toxicspikes",
                        "glare", "nuzzle", "stunspore", "yawn"}
        SCREENS = {"lightscreen", "reflect", "auroraveil"}
        PHAZING = {"whirlwind", "roar", "dragontail", "circlethrow", "haze"}
        PRIORITY = {"extremespeed", "bulletpunch", "machpunch", "iceshard",
                    "aquajet", "shadowsneak", "suckerpunch", "accelerock",
                    "quickattack", "grassyglide", "jetpunch", "firstimpression"}

        has_setup = bool(move_ids & SETUP)
        has_hazards = bool(move_ids & HAZARDS)
        has_removal = bool(move_ids & REMOVAL)
        has_recovery = bool(move_ids & RECOVERY)
        has_pivot = bool(move_ids & PIVOTS)
        has_status = bool(move_ids & STATUS_MOVES)
        has_screens = bool(move_ids & SCREENS)
        has_phazing = bool(move_ids & PHAZING)
        has_priority = bool(move_ids & PRIORITY)

        # Choice items
        is_choice = item in ("choiceband", "choicespecs", "choicescarf")
        is_scarf = item == "choicescarf"
        is_band = item == "choiceband"
        is_specs = item == "choicespecs"

        # Classify role
        # Setup sweeper
        if has_setup and best_offense >= 80:
            if atk > spa:
                setup_move = next((m for m in moves if _to_id(m) in SETUP), "")
                role_tags.append("Setup Sweeper")
                description_parts.append(f"Sets up with {setup_move} to sweep physically")
            else:
                setup_move = next((m for m in moves if _to_id(m) in SETUP), "")
                role_tags.append("Setup Sweeper")
                description_parts.append(f"Sets up with {setup_move} to sweep specially")

        # Wallbreaker
        if (is_band or is_specs) or (best_offense >= 120 and not has_setup):
            role_tags.append("Wallbreaker")
            if is_band:
                description_parts.append("Choice Band wallbreaker that punishes switches")
            elif is_specs:
                description_parts.append("Choice Specs wallbreaker with massive special power")
            else:
                description_parts.append("Raw power wallbreaker that pressures defensive cores")

        # Revenge killer / Scarfer
        if is_scarf:
            role_tags.append("Revenge Killer")
            description_parts.append("Choice Scarf user that picks off weakened threats")

        # Hazard setter
        if has_hazards:
            hazard_moves = [m for m in moves if _to_id(m) in HAZARDS]
            role_tags.append("Hazard Setter")
            description_parts.append(f"Sets {', '.join(hazard_moves)} to pressure opponents")

        # Hazard removal
        if has_removal:
            role_tags.append("Hazard Removal")
            removal_move = next((m for m in moves if _to_id(m) in REMOVAL), "")
            description_parts.append(f"Keeps hazards off the field with {removal_move}")

        # Defensive wall (thresholds based on base stat products: HP*Def or HP*SpD)
        if has_recovery and (phys_bulk >= 7000 or spec_bulk >= 7000):
            if phys_bulk > spec_bulk * 1.3:
                role_tags.append("Physical Wall")
                description_parts.append("Tanks physical hits and recovers HP")
            elif spec_bulk > phys_bulk * 1.3:
                role_tags.append("Special Wall")
                description_parts.append("Tanks special hits and recovers HP")
            else:
                role_tags.append("Mixed Wall")
                description_parts.append("Blanket check to both physical and special threats")

        # Pivot
        if has_pivot:
            pivot_move = next((m for m in moves if _to_id(m) in PIVOTS), "")
            role_tags.append("Pivot")
            description_parts.append(f"Uses {pivot_move} to maintain momentum")

        # Screens setter
        if has_screens:
            role_tags.append("Screens Setter")
            description_parts.append("Sets Light Screen/Reflect to support sweepers")

        # Status spreader
        if has_status and (phys_bulk >= 5000 or spec_bulk >= 5000):
            status_move = next((m for m in moves if _to_id(m) in STATUS_MOVES), "")
            role_tags.append("Status Spreader")
            description_parts.append(f"Cripples opponents with {status_move}")

        # Phaser
        if has_phazing:
            role_tags.append("Phaser")
            description_parts.append("Forces out setup sweepers and racks up hazard damage")

        # Priority user
        if has_priority and best_offense >= 100:
            priority_move = next((m for m in moves if _to_id(m) in PRIORITY), "")
            role_tags.append("Priority Attacker")
            description_parts.append(f"Picks off weakened foes with {priority_move}")

        # Fallback: offensive or defensive
        if not role_tags:
            if best_offense >= 100 or spe >= 100:
                role_tags.append("Offensive")
                description_parts.append("Applies offensive pressure")
            elif phys_bulk >= 6000 or spec_bulk >= 6000:
                role_tags.append("Defensive")
                description_parts.append("Provides defensive utility")
            else:
                role_tags.append("Support")
                description_parts.append("Provides utility and support")

        # Primary role is first tag
        primary_role = role_tags[0] if role_tags else "Unknown"

        return {
            "species": species,
            "role": primary_role,
            "tags": role_tags,
            "description": ". ".join(description_parts) + "." if description_parts else "",
            "types": types,
        }

    def _identify_win_conditions(
        self, team: list[dict], roles: list[dict]
    ) -> list[dict]:
        """Identify the team's primary and secondary win conditions."""
        win_cons = []
        role_tags_all = set()
        for r in roles:
            role_tags_all.update(r.get("tags", []))

        # Setup sweep
        sweepers = [r for r in roles if "Setup Sweeper" in r.get("tags", [])]
        if sweepers:
            names = [s["species"] for s in sweepers]
            win_cons.append({
                "type": "Setup Sweep",
                "priority": "primary" if len(sweepers) >= 2 else "secondary",
                "pokemon": names,
                "description": f"Set up with {' or '.join(names)} after removing checks, then sweep.",
            })

        # Wallbreaking + cleaning
        breakers = [r for r in roles if "Wallbreaker" in r.get("tags", [])]
        if breakers:
            names = [b["species"] for b in breakers]
            win_cons.append({
                "type": "Wallbreak & Clean",
                "priority": "primary" if not sweepers else "secondary",
                "pokemon": names,
                "description": f"Use {', '.join(names)} to weaken walls, then clean with faster Pokemon.",
            })

        # Hazard stacking
        hazard_setters = [r for r in roles if "Hazard Setter" in r.get("tags", [])]
        phasers = [r for r in roles if "Phaser" in r.get("tags", [])]
        if hazard_setters and (phasers or len(hazard_setters) >= 2):
            win_cons.append({
                "type": "Hazard Stack",
                "priority": "secondary",
                "pokemon": [h["species"] for h in hazard_setters],
                "description": "Stack hazards and force switches to wear down the opponent passively.",
            })

        # Defensive win (stall)
        walls = [r for r in roles
                 if any(t in r.get("tags", []) for t in ["Physical Wall", "Special Wall", "Mixed Wall"])]
        status_spreaders = [r for r in roles if "Status Spreader" in r.get("tags", [])]
        if len(walls) >= 3 or (len(walls) >= 2 and status_spreaders):
            win_cons.append({
                "type": "Defensive Attrition",
                "priority": "primary" if len(walls) >= 4 else "secondary",
                "pokemon": [w["species"] for w in walls],
                "description": "Wear down opponents through status, hazards, and passive damage.",
            })

        # Revenge killing chain
        scarfers = [r for r in roles if "Revenge Killer" in r.get("tags", [])]
        priority_users = [r for r in roles if "Priority Attacker" in r.get("tags", [])]
        if scarfers or priority_users:
            cleaners = scarfers + priority_users
            names = list({c["species"] for c in cleaners})
            win_cons.append({
                "type": "Revenge Kill & Clean",
                "priority": "secondary",
                "pokemon": names,
                "description": f"{', '.join(names)} can pick off weakened threats to close games.",
            })

        # Screens offense
        screeners = [r for r in roles if "Screens Setter" in r.get("tags", [])]
        if screeners and sweepers:
            win_cons.append({
                "type": "Screens Offense",
                "priority": "primary",
                "pokemon": [screeners[0]["species"]] + [s["species"] for s in sweepers],
                "description": f"Set screens with {screeners[0]['species']}, then set up safely behind them.",
            })

        if not win_cons:
            win_cons.append({
                "type": "General Offense",
                "priority": "primary",
                "pokemon": [r["species"] for r in roles],
                "description": "Apply pressure through type coverage and team synergy.",
            })

        # Sort so primary comes first
        win_cons.sort(key=lambda w: 0 if w["priority"] == "primary" else 1)
        return win_cons

    def _identify_synergies(
        self, team: list[dict], roles: list[dict]
    ) -> list[dict]:
        """Identify defensive cores and type synergies."""
        synergies = []

        # Check for complementary type pairings
        # Classic cores: Fire/Water/Grass, Dragon/Steel/Fairy, etc.
        CORES = [
            ({"Fire"}, {"Water"}, "Fire/Water core: Water covers Fire's Ground/Rock weakness, Fire covers Water's Grass weakness"),
            ({"Water"}, {"Grass"}, "Water/Grass core: Grass covers Water's Electric weakness, Water covers Grass's Fire weakness"),
            ({"Dragon"}, {"Steel"}, "Dragon/Steel core: Steel resists Dragon's Ice/Fairy weakness, Dragon resists Steel's Fire weakness"),
            ({"Dragon"}, {"Fairy"}, "Dragon/Fairy core: Fairy is immune to Dragon moves, Dragon resists many of Fairy's weaknesses"),
            ({"Ground"}, {"Flying"}, "Ground/Flying core: Flying is immune to Ground, Ground covers Rock/Electric"),
            ({"Fire"}, {"Grass"}, "Fire/Grass core: covers each other's weaknesses well"),
        ]

        type_map = {}
        for i, pkmn in enumerate(team):
            sp_id = _to_id(pkmn.get("species", ""))
            types = set(self._get_types(sp_id))
            type_map[pkmn.get("species", "")] = types

        found_cores = set()
        for species_a, types_a in type_map.items():
            for species_b, types_b in type_map.items():
                if species_a >= species_b:
                    continue
                for core_type_a, core_type_b, reason in CORES:
                    if (types_a & core_type_a and types_b & core_type_b) or \
                       (types_a & core_type_b and types_b & core_type_a):
                        core_key = tuple(sorted([species_a, species_b]))
                        if core_key not in found_cores:
                            found_cores.add(core_key)
                            synergies.append({
                                "pokemon": [species_a, species_b],
                                "type": "Defensive Core",
                                "description": reason,
                            })

        # VoltTurn core
        pivot_mons = [r["species"] for r in roles if "Pivot" in r.get("tags", [])]
        if len(pivot_mons) >= 2:
            synergies.append({
                "pokemon": pivot_mons[:3],
                "type": "VoltTurn Core",
                "description": f"{', '.join(pivot_mons[:3])} form a pivot chain to maintain momentum and generate free switches.",
            })

        # Hazard + spinblock synergy
        hazard_mons = [r["species"] for r in roles if "Hazard Setter" in r.get("tags", [])]
        ghost_mons = [pkmn.get("species", "") for pkmn in team
                      if "Ghost" in set(self._get_types(_to_id(pkmn.get("species", ""))))]
        if hazard_mons and ghost_mons:
            blockers = [g for g in ghost_mons if g not in hazard_mons]
            if blockers:
                synergies.append({
                    "pokemon": hazard_mons[:1] + blockers[:1],
                    "type": "Hazard Control",
                    "description": f"{blockers[0]} blocks Rapid Spin to preserve {hazard_mons[0]}'s hazards.",
                })

        return synergies

    def _build_game_plan(
        self, team: list[dict], roles: list[dict]
    ) -> dict[str, Any]:
        """Determine the team's game plan: lead, mid-game, end-game."""
        leads = []
        mid = []
        late = []

        for i, (pkmn, role) in enumerate(zip(team, roles)):
            tags = set(role.get("tags", []))
            species = role["species"]
            sp_id = _to_id(species)
            stats = self._get_base_stats(sp_id) or {}
            spe = stats.get("spe", 0)
            item = _to_id(pkmn.get("item", "") or "")

            # Lead candidates: hazard setters, screens, fast offensive
            if "Hazard Setter" in tags or "Screens Setter" in tags:
                leads.append({"species": species, "reason": "Sets up entry hazards/screens early"})
            elif spe >= 100 and ("Wallbreaker" in tags or "Offensive" in tags):
                leads.append({"species": species, "reason": "Fast offensive lead to apply early pressure"})

            # Mid-game: walls, pivots, breakers
            if any(t in tags for t in ["Physical Wall", "Special Wall", "Mixed Wall", "Pivot"]):
                mid.append({"species": species, "reason": "Controls pace, pivots, and absorbs hits"})
            elif "Wallbreaker" in tags:
                mid.append({"species": species, "reason": "Breaks down defensive cores mid-game"})

            # End-game: sweepers, scarfers, priority
            if "Setup Sweeper" in tags:
                late.append({"species": species, "reason": "Sweeps after checks are removed"})
            elif "Revenge Killer" in tags:
                late.append({"species": species, "reason": "Cleans up weakened threats with Scarf speed"})
            elif "Priority Attacker" in tags:
                late.append({"species": species, "reason": "Picks off low-HP foes with priority"})

        # If no lead identified, pick fastest or hazard mon
        if not leads:
            fastest = max(
                range(len(team)),
                key=lambda i: (self._get_base_stats(_to_id(team[i].get("species", ""))) or {}).get("spe", 0),
            )
            leads.append({
                "species": team[fastest].get("species", ""),
                "reason": "Fastest team member, applies early pressure",
            })

        return {
            "lead": leads[:2],
            "mid_game": mid[:3],
            "end_game": late[:3],
        }

    def _generate_summary(
        self,
        roles: list[dict],
        win_conditions: list[dict],
        synergies: list[dict],
        game_plan: dict,
    ) -> str:
        """Generate a human-readable strategy summary."""
        parts = []

        # Team composition summary
        role_counts = {}
        for r in roles:
            primary = r.get("role", "Unknown")
            role_counts[primary] = role_counts.get(primary, 0) + 1

        role_str = ", ".join(f"{count} {role}{'s' if count > 1 else ''}" for role, count in role_counts.items())
        parts.append(f"This team features {role_str}.")

        # Primary win condition
        primary_wc = next((w for w in win_conditions if w["priority"] == "primary"), None)
        if primary_wc:
            parts.append(f"Primary win condition: {primary_wc['description']}")

        # Secondary win conditions
        secondary_wcs = [w for w in win_conditions if w["priority"] == "secondary"]
        if secondary_wcs:
            parts.append(f"Backup plan: {secondary_wcs[0]['description']}")

        # Game plan
        if game_plan.get("lead"):
            lead_names = [l["species"] for l in game_plan["lead"]]
            parts.append(f"Lead with {' or '.join(lead_names)} to set the pace early.")

        if game_plan.get("end_game"):
            closer_names = [e["species"] for e in game_plan["end_game"]]
            parts.append(f"Close out games with {' or '.join(closer_names)}.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Pairwise Matchup Matrix (Head-to-Head)
    # ------------------------------------------------------------------

    def pairwise_matchup_matrix(
        self, team1: list[dict], team2: list[dict], feature_extractor=None
    ) -> list[dict]:
        """Compute 6x6 damage grid between two teams.

        For each (attacker, defender) pair, finds the best move and estimated
        damage %.  Returns a flat list of matchup entries.

        Args:
            team1: Attacker team (list of PokemonSet dicts).
            team2: Defender team (list of PokemonSet dicts).
            feature_extractor: Optional FeatureExtractor with precompute_team_data.

        Returns:
            List of dicts with keys: atk, def_, atk_move, atk_dmg, def_move, def_dmg.
        """
        if feature_extractor is None:
            from ..data.features import FeatureExtractor
            feature_extractor = FeatureExtractor(self.pokemon_data)

        t1_data = feature_extractor.precompute_team_data(team1)
        t2_data = feature_extractor.precompute_team_data(team2)

        results = []
        for i, (atk_set, atk_pre) in enumerate(zip(team1, t1_data)):
            for j, (def_set, def_pre) in enumerate(zip(team2, t2_data)):
                # Best move from attacker -> defender
                atk_best_dmg = 0.0
                atk_best_move = ""
                for md in atk_pre.get("moves_data", []):
                    if md.get("category") == "Status":
                        continue
                    dmg = estimate_damage_pct(atk_pre, def_pre, md)
                    if dmg > atk_best_dmg:
                        atk_best_dmg = dmg
                        atk_best_move = md.get("name", "")

                # Best move from defender -> attacker
                def_best_dmg = 0.0
                def_best_move = ""
                for md in def_pre.get("moves_data", []):
                    if md.get("category") == "Status":
                        continue
                    dmg = estimate_damage_pct(def_pre, atk_pre, md)
                    if dmg > def_best_dmg:
                        def_best_dmg = dmg
                        def_best_move = md.get("name", "")

                results.append({
                    "atk": atk_set.get("species", f"Mon {i+1}"),
                    "def_": def_set.get("species", f"Mon {j+1}"),
                    "atk_move": atk_best_move,
                    "atk_dmg": round(float(atk_best_dmg * 100), 1),
                    "def_move": def_best_move,
                    "def_dmg": round(float(def_best_dmg * 100), 1),
                })
        return results

    # ------------------------------------------------------------------
    # Counter / Check Finder
    # ------------------------------------------------------------------

    def find_counters(
        self, target: dict, pokemon_pool: list[dict], n: int = 10,
        feature_extractor=None,
    ) -> dict[str, list[dict]]:
        """Find counters and checks for a target Pokemon from the pool.

        Counter: takes < 33% from target AND deals > 40% to target.
        Check: deals > 50% to target (threatens KO but may not switch in safely).

        Returns dict with 'counters' and 'checks' lists.
        """
        if feature_extractor is None:
            from ..data.features import FeatureExtractor
            feature_extractor = FeatureExtractor(self.pokemon_data)

        target_data = feature_extractor.precompute_team_data([target])
        if not target_data:
            return {"counters": [], "checks": []}
        target_pre = target_data[0]

        # Deduplicate pool by species
        seen = set()
        candidates = []
        for p in pokemon_pool:
            sp = _to_id(p.get("species", ""))
            if sp and sp not in seen:
                seen.add(sp)
                candidates.append(p)

        scored = []
        for cand in candidates:
            cand_data = feature_extractor.precompute_team_data([cand])
            if not cand_data:
                continue
            cand_pre = cand_data[0]

            dmg_to_target = estimate_best_move_damage(cand_pre, target_pre)
            dmg_from_target = estimate_best_move_damage(target_pre, cand_pre)

            # Find best move names
            best_move_to = ""
            best_val = 0.0
            for md in cand_pre.get("moves_data", []):
                if md.get("category") == "Status":
                    continue
                d = estimate_damage_pct(cand_pre, target_pre, md)
                if d > best_val:
                    best_val = d
                    best_move_to = md.get("name", "")

            scored.append({
                "species": cand.get("species", ""),
                "dmg_to_target": float(dmg_to_target),
                "dmg_from_target": float(dmg_from_target),
                "best_move": best_move_to,
            })

        counters = []
        checks = []
        for s in scored:
            if s["dmg_from_target"] < 0.33 and s["dmg_to_target"] > 0.40:
                counters.append({
                    "species": s["species"],
                    "best_move": s["best_move"],
                    "dmg_to_target": round(s["dmg_to_target"] * 100, 1),
                    "dmg_from_target": round(s["dmg_from_target"] * 100, 1),
                    "sprite": f"https://play.pokemonshowdown.com/sprites/gen5/{_to_id(s['species'])}.png",
                })
            elif s["dmg_to_target"] > 0.50:
                checks.append({
                    "species": s["species"],
                    "best_move": s["best_move"],
                    "dmg_to_target": round(s["dmg_to_target"] * 100, 1),
                    "dmg_from_target": round(s["dmg_from_target"] * 100, 1),
                    "sprite": f"https://play.pokemonshowdown.com/sprites/gen5/{_to_id(s['species'])}.png",
                })

        # Sort counters by dmg_to_target desc, checks by dmg_to_target desc
        counters.sort(key=lambda x: -x["dmg_to_target"])
        checks.sort(key=lambda x: -x["dmg_to_target"])

        return {
            "counters": counters[:n],
            "checks": checks[:n],
        }

    # ------------------------------------------------------------------
    # Slot Replacement Suggestions
    # ------------------------------------------------------------------

    def suggest_slot_replacement(
        self, team: list[dict], slot_idx: int,
        pool: list[dict], evaluator, meta_teams: list[list[dict]],
        top_n: int = 8,
    ) -> list[dict]:
        """Suggest replacements for a team slot.

        Removes the mon at slot_idx, tries top candidates from pool,
        evaluates each, returns best by fitness with delta.

        Args:
            team: Current 6-mon team.
            slot_idx: Which slot to replace (0-5).
            pool: Pokemon pool to draw candidates from.
            evaluator: TeamEvaluator instance (fast_mode recommended).
            meta_teams: Meta teams for alignment scoring.
            top_n: Number of suggestions to return.

        Returns:
            List of dicts with species, fitness, delta, sprite.
        """
        if slot_idx < 0 or slot_idx >= len(team):
            return []

        # Score current team
        current_fitness = evaluator.evaluate(team)

        # Get existing species to avoid duplicates
        existing = {_to_id(p.get("species", "")) for p in team if p}

        # Deduplicate pool by species, take top 50 by frequency
        species_count = {}
        species_best = {}
        for p in pool:
            sp = _to_id(p.get("species", ""))
            if sp and sp not in existing:
                species_count[sp] = species_count.get(sp, 0) + 1
                if sp not in species_best:
                    species_best[sp] = p

        # Sort by usage (frequency in pool)
        top_species = sorted(species_count.keys(), key=lambda s: -species_count[s])[:50]

        results = []
        for sp in top_species:
            cand = species_best[sp]
            test_team = list(team)
            test_team[slot_idx] = cand
            fitness = evaluator.evaluate(test_team)
            delta = fitness - current_fitness
            results.append({
                "species": cand.get("species", ""),
                "fitness": round(float(fitness), 4),
                "delta": round(float(delta), 4),
                "sprite": f"https://play.pokemonshowdown.com/sprites/gen5/{_to_id(cand.get('species', ''))}.png",
            })

        results.sort(key=lambda x: -x["fitness"])
        return results[:top_n]

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
