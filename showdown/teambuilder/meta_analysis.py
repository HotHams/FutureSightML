"""Metagame analysis: builds a model of the current meta for team evaluation."""

import json
import logging
import random
import re
from typing import Any

from ..data.database import Database
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

    def __init__(self, db: Database):
        self.db = db
        self.stats_scraper = StatsScraper(db)

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
        return sets[0] if sets else {
            "species": species, "ability": None, "item": None,
            "moves": [], "nature": None, "tera_type": None,
        }

    def build_pokemon_sets(
        self, species: str, data: dict, max_sets: int = 4
    ) -> list[dict]:
        """Generate multiple viable sets for a Pokemon from usage data.

        Produces distinct sets by varying items, abilities, and move combinations.
        """
        abilities = data.get("abilities", {})
        items = data.get("items", {})
        moves_data = data.get("moves", {})
        spreads = data.get("spreads", {})

        top_abilities = sorted(abilities.items(), key=lambda x: x[1], reverse=True)[:3]
        top_items = sorted(items.items(), key=lambda x: x[1], reverse=True)[:4]
        top_moves = sorted(moves_data.items(), key=lambda x: x[1], reverse=True)[:10]

        # Extract nature from top spread
        nature = None
        if spreads:
            top_spread = max(spreads, key=spreads.get)
            if ":" in top_spread:
                nature = top_spread.split(":")[0]

        sets = []
        seen_keys = set()

        # Generate sets by combining top items with top abilities
        for item, _ in (top_items or [(None, 0)]):
            for ability, _ in (top_abilities or [(None, 0)]):
                if len(sets) >= max_sets:
                    break
                # Pick top 4 moves
                moveset = [m for m, _ in top_moves[:4]]
                key = (ability, item, tuple(sorted(moveset)))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                sets.append({
                    "species": species,
                    "ability": ability,
                    "item": item,
                    "moves": moveset,
                    "nature": nature,
                    "tera_type": None,
                })

        # Also generate a set with alternate moves (5-8 instead of 1-4)
        if len(top_moves) > 4 and len(sets) < max_sets:
            alt_moves = [m for m, _ in top_moves[2:6]]  # overlap by 2
            ability = top_abilities[0][0] if top_abilities else None
            item = top_items[0][0] if top_items else None
            key = (ability, item, tuple(sorted(alt_moves)))
            if key not in seen_keys:
                sets.append({
                    "species": species,
                    "ability": ability,
                    "item": item,
                    "moves": alt_moves,
                    "nature": nature,
                    "tera_type": None,
                })

        return sets if sets else [{
            "species": species, "ability": None, "item": None,
            "moves": [m for m, _ in top_moves[:4]], "nature": nature,
            "tera_type": None,
        }]

    def build_full_pokemon_pool(
        self, usage: dict[str, dict], top_n: int = 80, sets_per_pokemon: int = 4
    ) -> list[dict]:
        """Build a rich pool of Pokemon sets from usage data.

        Returns hundreds of distinct (species, moveset, item, ability) combinations.
        """
        sorted_pokemon = sorted(
            usage.items(),
            key=lambda x: x[1]["usage_pct"],
            reverse=True,
        )[:top_n]

        pool = []
        for species_id, data in sorted_pokemon:
            sets = self.build_pokemon_sets(species_id, data, max_sets=sets_per_pokemon)
            pool.extend(sets)

        log.info(
            "Built Pokemon pool: %d sets across %d species",
            len(pool), len(set(p["species"] for p in pool)),
        )
        return pool

    async def _teams_from_replays(
        self, format_id: str, n_teams: int
    ) -> list[list[dict]]:
        """Fallback: extract meta teams from stored replays."""
        battles = await self.db.get_training_battles(
            format_id, min_rating=1700, limit=n_teams * 2
        )

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
