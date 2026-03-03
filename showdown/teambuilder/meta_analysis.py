"""Metagame analysis: builds a model of the current meta for team evaluation."""

import json
import logging
import random
import re
from typing import Any

from ..data.database import Database
from ..data.pokemon_data import PokemonDataLoader
from ..scraper.stats_scraper import StatsScraper

log = logging.getLogger("showdown.teambuilder.meta_analysis")


def _to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class MetaAnalyzer:
    """Analyze the metagame and generate representative opponent teams.

    This is critical for the team builder: we evaluate candidate teams
    against a distribution of meta teams, so the distribution must be
    representative of what you'll actually face on the ladder.
    """

    def __init__(self, db: Database, pokemon_data: PokemonDataLoader | None = None, gen: int = 9):
        self.db = db
        self.pokemon_data = pokemon_data
        self.gen = gen
        self.stats_scraper = StatsScraper(db)

    def _validate_ability(self, species_id: str, ability: str | None) -> str | None:
        """Validate that an ability belongs to the species; return corrected ability."""
        if not self.pokemon_data:
            return ability
        pkmn_entry = self.pokemon_data.get_pokemon(species_id)
        if not pkmn_entry:
            return ability
        valid_abilities = {_to_id(a) for a in pkmn_entry.get("abilities", {}).values()}
        if not valid_abilities:
            return ability
        if ability and _to_id(ability) in valid_abilities:
            return ability
        # Fall back to first valid ability
        return list(pkmn_entry.get("abilities", {}).values())[0]

    def _pad_moveset(
        self, species_id: str, moveset: list[str], target: int = 4, gen: int = 9
    ) -> list[str]:
        """Pad a moveset to `target` moves using type-appropriate filler.

        Respects generation boundaries and per-Pokemon learnability:
        - Only considers moves that existed in the given gen
        - For Gen 3+, validates learnability via Showdown learnset data
        - Prefers STAB damaging moves the Pokemon can actually learn

        Priority:
        1. Moves already in the list (no-op if >= target)
        2. Learnable STAB damaging moves for the species' types
        3. Common competitive utility moves legal in this gen
        """
        if len(moveset) >= target:
            return moveset[:target]

        moveset = list(moveset)  # don't mutate caller's list
        used = {_to_id(m) for m in moveset}

        # Try STAB damaging moves from the move database
        if self.pokemon_data:
            pkmn_entry = self.pokemon_data.get_pokemon(species_id)
            types = pkmn_entry.get("types", []) if pkmn_entry else []

            # Collect candidate moves: must exist in gen AND be learnable
            type_moves = []
            for move_id, move_data in self.pokemon_data.moves.items():
                if move_id in used:
                    continue
                if not self.pokemon_data.move_exists_in_gen(move_id, gen):
                    continue
                if not self.pokemon_data.can_learn_move(species_id, move_id, gen):
                    continue
                if move_data.get("category") in ("Physical", "Special"):
                    bp = move_data.get("basePower", 0) or 0
                    mtype = move_data.get("type", "")
                    stab = 1 if mtype in types else 0
                    type_moves.append((move_id, stab, bp))
            # Sort: STAB first, then highest base power
            type_moves.sort(key=lambda x: (x[1], x[2]), reverse=True)
            for mid, _, _ in type_moves:
                if mid not in used:
                    moveset.append(mid)
                    used.add(mid)
                if len(moveset) >= target:
                    return moveset[:target]

        # Last resort: generic utility moves (filtered by gen)
        fillers = [
            "protect", "toxic", "rest", "substitute", "return",
            "bodyslam", "thunderwave", "icebeam", "earthquake", "surf",
            "psychic", "flamethrower", "thunderbolt", "shadowball", "swordsdance",
        ]
        for f in fillers:
            if f not in used:
                if self.pokemon_data and not self.pokemon_data.move_exists_in_gen(f, gen):
                    continue
                moveset.append(f)
                used.add(f)
            if len(moveset) >= target:
                break

        return moveset[:target]

    async def build_meta_teams(
        self,
        format_id: str,
        year_month: str | None = None,
        n_teams: int = 50,
        rating_threshold: int = 1825,
    ) -> list[list[dict]]:
        """Generate representative meta teams from usage statistics.

        Uses teammate correlation data to build realistic teams that
        reflect the current metagame.

        Returns: list of teams, each team is a list of pokemon set dicts.
        """
        if year_month is None:
            year_month = await self.db.get_latest_usage_month(format_id)
            if year_month is None:
                log.warning("No usage stats available for %s", format_id)
                return await self._teams_from_replays(format_id, n_teams)

        raw_data = await self.db.get_usage_stats(format_id, year_month, rating_threshold)
        # Fall back to lower rating thresholds if high-level stats unavailable
        if raw_data is None:
            for fallback_rating in [1760, 1630, 1500, 0]:
                if fallback_rating >= rating_threshold:
                    continue
                raw_data = await self.db.get_usage_stats(format_id, year_month, fallback_rating)
                if raw_data is not None:
                    log.info("Using fallback rating threshold %d for %s", fallback_rating, format_id)
                    break
        if raw_data is None:
            log.warning("No stats for %s %s", format_id, year_month)
            return await self._teams_from_replays(format_id, n_teams)

        usage = self.stats_scraper.parse_usage_data(raw_data)
        return self._generate_teams_from_usage(usage, n_teams)

    def _generate_teams_from_usage(
        self,
        usage: dict[str, dict],
        n_teams: int,
    ) -> list[list[dict]]:
        """Generate teams by sampling Pokemon weighted by usage and teammate synergy."""
        # Sort by usage
        sorted_pokemon = sorted(
            usage.items(),
            key=lambda x: x[1]["usage_pct"],
            reverse=True,
        )

        # Top 50 Pokemon form the "meta pool"
        meta_pool = sorted_pokemon[:50]
        pool_species = [sp for sp, _ in meta_pool]
        usage_weights = [data["usage_pct"] for _, data in meta_pool]
        total_weight = sum(usage_weights)
        if total_weight > 0:
            usage_probs = [w / total_weight for w in usage_weights]
        else:
            usage_probs = [1.0 / len(meta_pool)] * len(meta_pool)

        teams = []
        for _ in range(n_teams):
            team = self._sample_team(pool_species, usage_probs, usage, meta_pool)
            teams.append(team)

        return teams

    def _sample_team(
        self,
        pool_species: list[str],
        usage_probs: list[float],
        usage: dict[str, dict],
        meta_pool: list[tuple[str, dict]],
    ) -> list[dict]:
        """Sample a single team using usage-weighted selection with teammate bias."""
        team = []
        used_species = set()

        # Pick lead based on usage
        lead_idx = random.choices(range(len(pool_species)), weights=usage_probs, k=1)[0]
        lead_species = pool_species[lead_idx]
        lead_set = self._build_pokemon_set(lead_species, usage.get(lead_species, {}))
        team.append(lead_set)
        used_species.add(lead_species)

        # Pick remaining 5 Pokemon, biased by teammate correlation
        for _ in range(5):
            if len(used_species) >= len(pool_species):
                break

            # Build weights combining usage and teammate synergy
            weights = []
            for i, sp in enumerate(pool_species):
                if sp in used_species:
                    weights.append(0.0)
                    continue

                base_w = usage_probs[i]
                # Add teammate bonus from each team member
                teammate_bonus = 0.0
                sp_data = usage.get(sp, {})
                teammates = sp_data.get("teammates", {})
                for team_member in team:
                    tm_id = _to_id(team_member["species"])
                    teammate_bonus += teammates.get(tm_id, 0) / 100.0

                weights.append(base_w * (1.0 + teammate_bonus))

            total = sum(weights)
            if total == 0:
                break
            weights = [w / total for w in weights]

            idx = random.choices(range(len(pool_species)), weights=weights, k=1)[0]
            species = pool_species[idx]
            pkmn_set = self._build_pokemon_set(species, usage.get(species, {}))
            team.append(pkmn_set)
            used_species.add(species)

        return team

    def _build_pokemon_set(self, species: str, data: dict) -> dict:
        """Build a single Pokemon set from usage statistics (most popular)."""
        sets = self.build_pokemon_sets(species, data, max_sets=1)
        if sets:
            return sets[0]
        # Fallback: pad with type-appropriate moves
        return {
            "species": species, "ability": self._validate_ability(species, None),
            "item": None, "moves": self._pad_moveset(species, [], gen=self.gen),
            "nature": None, "tera_type": None,
        }

    @staticmethod
    def _parse_spread(spread_str: str) -> tuple[str, dict[str, int]]:
        """Parse a Smogon spread string like 'Adamant:252/0/4/0/0/252'.

        Returns (nature, {hp: N, atk: N, def: N, spa: N, spd: N, spe: N}).
        """
        stat_keys = ["hp", "atk", "def", "spa", "spd", "spe"]
        if ":" not in spread_str:
            return ("Hardy", {})
        nature, evs_part = spread_str.split(":", 1)
        ev_vals = evs_part.split("/")
        evs = {}
        for i, val in enumerate(ev_vals):
            if i < len(stat_keys):
                v = int(val)
                if v > 0:
                    evs[stat_keys[i]] = v
        return (nature, evs)

    def build_pokemon_sets(
        self, species: str, data: dict, max_sets: int = 4
    ) -> list[dict]:
        """Generate multiple viable sets for a Pokemon from usage data.

        Produces distinct sets by varying items, abilities, and move combinations.
        Each set includes proper EV spread and nature from Smogon stats.
        """
        abilities = data.get("abilities", {})
        items = data.get("items", {})
        moves_data = data.get("moves", {})
        spreads = data.get("spreads", {})

        top_abilities = sorted(abilities.items(), key=lambda x: x[1], reverse=True)[:3]
        top_items = sorted(items.items(), key=lambda x: x[1], reverse=True)[:4]
        top_moves = sorted(moves_data.items(), key=lambda x: x[1], reverse=True)[:10]
        top_spreads = sorted(spreads.items(), key=lambda x: x[1], reverse=True)[:4]

        # Parse top spread for nature + EVs
        nature = None
        evs = {}
        if top_spreads:
            nature, evs = self._parse_spread(top_spreads[0][0])

        sets = []
        seen_keys = set()

        # Primary set: top item + top ability + top 4 moves + top spread
        primary_moveset = self._pad_moveset(species, [m for m, _ in top_moves[:4]], gen=self.gen)
        primary_ability = self._validate_ability(species, top_abilities[0][0] if top_abilities else None)
        primary_item = top_items[0][0] if top_items else None
        sets.append({
            "species": species,
            "ability": primary_ability,
            "item": primary_item,
            "moves": primary_moveset,
            "nature": nature,
            "evs": evs,
            "tera_type": None,
        })
        seen_keys.add((primary_ability, primary_item, tuple(sorted(primary_moveset))))

        # Variant sets: different items suggest different roles
        for item, _ in top_items[1:]:
            if len(sets) >= max_sets:
                break
            # Different item often means different spread
            spread_idx = min(len(sets), len(top_spreads) - 1)
            set_nature, set_evs = (
                self._parse_spread(top_spreads[spread_idx][0])
                if spread_idx >= 0 and top_spreads
                else (nature, evs)
            )
            ability = self._validate_ability(species, top_abilities[0][0] if top_abilities else None)
            moveset = primary_moveset[:]
            key = (ability, item, tuple(sorted(moveset)))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            sets.append({
                "species": species,
                "ability": ability,
                "item": item,
                "moves": moveset,
                "nature": set_nature,
                "evs": set_evs,
                "tera_type": None,
            })

        # Alternate moveset variant (moves 3-6)
        if len(top_moves) > 4 and len(sets) < max_sets:
            alt_moves = self._pad_moveset(species, [m for m, _ in top_moves[2:6]], gen=self.gen)
            ability = primary_ability
            item = primary_item
            key = (ability, item, tuple(sorted(alt_moves)))
            if key not in seen_keys:
                seen_keys.add(key)
                sets.append({
                    "species": species,
                    "ability": ability,
                    "item": item,
                    "moves": alt_moves,
                    "nature": nature,
                    "evs": evs,
                    "tera_type": None,
                })

        # Alternate ability variant
        if len(top_abilities) > 1 and len(sets) < max_sets:
            alt_ability = self._validate_ability(species, top_abilities[1][0])
            key = (alt_ability, primary_item, tuple(sorted(primary_moveset)))
            if key not in seen_keys:
                seen_keys.add(key)
                sets.append({
                    "species": species,
                    "ability": alt_ability,
                    "item": primary_item,
                    "moves": primary_moveset,
                    "nature": nature,
                    "evs": evs,
                    "tera_type": None,
                })

        fallback_ability = self._validate_ability(species, None)
        fallback_moves = self._pad_moveset(species, [m for m, _ in top_moves[:4]], gen=self.gen)
        return sets if sets else [{
            "species": species, "ability": fallback_ability, "item": None,
            "moves": fallback_moves, "nature": nature,
            "evs": evs, "tera_type": None,
        }]

    def build_full_pokemon_pool(
        self, usage: dict[str, dict], top_n: int = 80, sets_per_pokemon: int = 4
    ) -> list[dict]:
        """Build a rich pool of Pokemon sets from usage data.

        Returns hundreds of distinct (species, moveset, item, ability) combinations.
        All sets are guaranteed to have 4 moves.
        """
        sorted_pokemon = sorted(
            usage.items(),
            key=lambda x: x[1]["usage_pct"],
            reverse=True,
        )[:top_n]

        pool = []
        global_seen = set()
        for species_id, data in sorted_pokemon:
            sets = self.build_pokemon_sets(species_id, data, max_sets=sets_per_pokemon)
            for s in sets:
                key = (_to_id(s.get("species", "")),
                       _to_id(s.get("ability") or ""),
                       _to_id(s.get("item") or ""),
                       tuple(sorted(_to_id(m) for m in s.get("moves", []))))
                if key not in global_seen:
                    global_seen.add(key)
                    pool.append(s)

        # Enforce 4-move minimum
        before = len(pool)
        pool = [p for p in pool if len([m for m in p.get("moves", []) if m]) >= 4]
        if before != len(pool):
            log.info("Dropped %d sets with <4 moves from usage pool", before - len(pool))

        log.info(
            "Built Pokemon pool: %d sets across %d species",
            len(pool), len(set(p["species"] for p in pool)),
        )
        return pool

    async def _teams_from_replays(
        self, format_id: str, n_teams: int
    ) -> list[list[dict]]:
        """Fallback: extract meta teams from stored replays."""
        # Try progressively lower rating thresholds
        battles = []
        for min_r in [1700, 1500, 1200, 0]:
            battles = await self.db.get_training_battles(
                format_id, min_rating=min_r, limit=n_teams * 2
            )
            if len(battles) >= n_teams:
                break

        teams = []
        for battle in battles:
            # Use the winning team
            if battle["winner"] == 1:
                teams.append(battle["team1"])
            else:
                teams.append(battle["team2"])
            if len(teams) >= n_teams:
                break

        return teams

    async def build_pool_from_replays(
        self, format_id: str, top_n: int = 80, sets_per_pokemon: int = 4
    ) -> list[dict]:
        """Build a Pokemon pool by mining replay data for common sets.

        Analyzes winning teams from high-rated replays to find the most
        popular species, moves, items, and abilities — then builds
        distinct sets for each species.
        """
        from collections import Counter

        battles = await self.db.get_training_battles(
            format_id, min_rating=1500, limit=5000
        )

        if not battles:
            battles = await self.db.get_training_battles(
                format_id, min_rating=0, limit=5000
            )

        if not battles:
            log.warning("No replay data for %s, cannot build pool", format_id)
            return []

        # Collect all Pokemon appearances from winning teams
        species_count: Counter = Counter()
        species_moves: dict[str, Counter] = {}
        species_items: dict[str, Counter] = {}
        species_abilities: dict[str, Counter] = {}

        for battle in battles:
            # Prefer winning team, but use both for larger sample
            for team_key in ["team1", "team2"]:
                team = battle.get(team_key, [])
                if not isinstance(team, list):
                    continue
                for pkmn in team:
                    if not isinstance(pkmn, dict):
                        continue
                    sp = pkmn.get("species", "")
                    if not sp:
                        continue
                    sp_id = _to_id(sp)
                    species_count[sp_id] += 1

                    if sp_id not in species_moves:
                        species_moves[sp_id] = Counter()
                        species_items[sp_id] = Counter()
                        species_abilities[sp_id] = Counter()

                    for m in pkmn.get("moves", []):
                        if m:
                            species_moves[sp_id][m] += 1
                    item = pkmn.get("item")
                    if item:
                        species_items[sp_id][item] += 1
                    ability = pkmn.get("ability")
                    if ability:
                        species_abilities[sp_id][ability] += 1

        # Take top N species by usage
        top_species = species_count.most_common(top_n)

        # Frequency-based filtering: remove extreme outliers (<1% of median)
        if top_species:
            counts = [c for _, c in top_species]
            median_count = sorted(counts)[len(counts) // 2]
            min_count = max(1, int(median_count * 0.01))
            top_species = [(sp, c) for sp, c in top_species if c >= min_count]

        pool = []
        global_seen = set()

        for sp_id, count in top_species:
            all_moves_for_species = [m for m, _ in species_moves[sp_id].most_common(20)]
            top_moves_list = all_moves_for_species[:10]
            top_items_list = [it for it, _ in species_items[sp_id].most_common(4)]
            top_abilities_list = [ab for ab, _ in species_abilities[sp_id].most_common(3)]

            sets_made = 0

            # Generate distinct sets by varying items and abilities
            for item in (top_items_list or [None]):
                for ability in (top_abilities_list or [None]):
                    if sets_made >= sets_per_pokemon:
                        break
                    # Start with top 4 observed moves, pad from all observed,
                    # then fall back to type-appropriate moves from pokemon_data
                    raw_moveset = list(top_moves_list[:4])
                    # First pad from ALL observed moves for this species
                    for m in all_moves_for_species:
                        if len(raw_moveset) >= 4:
                            break
                        if m not in raw_moveset:
                            raw_moveset.append(m)
                    # Then pad from pokemon_data (STAB moves, utilities)
                    moveset = self._pad_moveset(sp_id, raw_moveset, target=4, gen=self.gen)
                    if len(moveset) < 4:
                        continue  # skip if even padding couldn't reach 4
                    ability = self._validate_ability(sp_id, ability)
                    key = (_to_id(sp_id), _to_id(ability or ""), _to_id(item or ""), tuple(sorted(_to_id(m) for m in moveset)))
                    if key in global_seen:
                        continue
                    global_seen.add(key)
                    pool.append({
                        "species": sp_id,
                        "ability": ability,
                        "item": item,
                        "moves": moveset,
                        "nature": None,
                        "tera_type": None,
                    })
                    sets_made += 1

            # Add an alternate moveset if available
            if len(top_moves_list) > 4 and sets_made < sets_per_pokemon:
                raw_alt = list(top_moves_list[2:6])
                for m in all_moves_for_species:
                    if len(raw_alt) >= 4:
                        break
                    if m not in raw_alt:
                        raw_alt.append(m)
                alt_moves = self._pad_moveset(sp_id, raw_alt, target=4, gen=self.gen)
                if len(alt_moves) >= 4:
                    ability = self._validate_ability(sp_id, top_abilities_list[0] if top_abilities_list else None)
                    item = top_items_list[0] if top_items_list else None
                    key = (_to_id(sp_id), _to_id(ability or ""), _to_id(item or ""), tuple(sorted(_to_id(m) for m in alt_moves)))
                    if key not in global_seen:
                        pool.append({
                            "species": sp_id,
                            "ability": ability,
                            "item": item,
                            "moves": alt_moves,
                            "nature": None,
                            "tera_type": None,
                        })

        # Final pool validation: enforce 4-move minimum
        pool = [p for p in pool if len([m for m in p.get("moves", []) if m]) >= 4]

        log.info(
            "Built Pokemon pool from replays: %d sets across %d species",
            len(pool), len(set(p["species"] for p in pool)),
        )
        return pool

    async def get_meta_summary(
        self,
        format_id: str,
        year_month: str | None = None,
        top_n: int = 30,
    ) -> dict[str, Any]:
        """Get a summary of the current metagame for a format."""
        if year_month is None:
            year_month = await self.db.get_latest_usage_month(format_id)

        if year_month is None:
            return {"format": format_id, "pokemon": [], "error": "No data available"}

        raw_data = await self.db.get_usage_stats(format_id, year_month)
        if raw_data is None:
            return {"format": format_id, "pokemon": [], "error": "No stats"}

        usage = self.stats_scraper.parse_usage_data(raw_data)
        top = self.stats_scraper.get_top_pokemon(usage, top_n)

        return {
            "format": format_id,
            "year_month": year_month,
            "total_pokemon": len(usage),
            "pokemon": top,
        }
