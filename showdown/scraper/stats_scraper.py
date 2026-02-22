"""Scraper for Smogon monthly usage statistics."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any

import aiohttp

from ..data.database import Database

log = logging.getLogger("showdown.scraper.stats_scraper")


class StatsScraper:
    """Scrape monthly usage statistics from Smogon's stats page.

    Stats are available at:
      https://www.smogon.com/stats/{YYYY-MM}/chaos/{format}-{rating}.json

    The 'chaos' JSON format contains rich data:
    - Pokemon usage percentages
    - Move usage per Pokemon
    - Item usage per Pokemon
    - Ability usage per Pokemon
    - Spread (nature + EVs) usage per Pokemon
    - Teammate correlation data
    - Checks & counters data
    """

    def __init__(
        self,
        db: Database,
        base_url: str = "https://www.smogon.com/stats",
        request_delay_ms: int = 500,
    ):
        self.db = db
        self.base_url = base_url.rstrip("/")
        self.request_delay = request_delay_ms / 1000.0
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "PokemonShowdownML/0.1 (research)",
                },
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def scrape_format(
        self,
        format_id: str,
        months: list[str] | None = None,
        rating_thresholds: list[int] | None = None,
    ) -> int:
        """Scrape usage stats for a format across multiple months.

        Args:
            format_id: Showdown format string (e.g. 'gen9ou')
            months: List of 'YYYY-MM' strings. If None, scrapes last 6 months.
            rating_thresholds: Rating cutoffs (e.g. [0, 1500, 1695, 1825]).
                              If None, uses [1825] (high-level play).

        Returns:
            Number of stat files successfully scraped.
        """
        if months is None:
            months = self._recent_months(6)
        if rating_thresholds is None:
            rating_thresholds = [1825]

        count = 0
        for month in months:
            for rating in rating_thresholds:
                success = await self._scrape_single(format_id, month, rating)
                if success:
                    count += 1
                await asyncio.sleep(self.request_delay)

        await self.db.commit()
        log.info("Scraped %d stat files for %s", count, format_id)
        return count

    async def _scrape_single(
        self, format_id: str, year_month: str, rating: int
    ) -> bool:
        """Scrape a single stats file."""
        url = f"{self.base_url}/{year_month}/chaos/{format_id}-{rating}.json"

        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status == 404:
                    log.debug("Stats not found: %s", url)
                    return False
                resp.raise_for_status()
                text = await resp.text()
                data = json.loads(text)
        except Exception as e:
            log.warning("Failed to fetch stats %s: %s", url, e)
            return False

        info = data.get("info", {})
        total_battles = info.get("number of battles", 0)

        await self.db.insert_usage_stats(
            format_id=format_id,
            year_month=year_month,
            rating_threshold=rating,
            total_battles=total_battles,
            data_json=text,
        )

        pokemon_count = len(data.get("data", {}))
        log.info(
            "Scraped stats: %s %s (rating>=%d) - %d battles, %d Pokemon",
            format_id, year_month, rating, total_battles, pokemon_count,
        )
        return True

    def parse_usage_data(self, raw_data: dict) -> dict[str, dict[str, Any]]:
        """Parse the chaos JSON format into a structured per-Pokemon dict.

        Returns:
            Dict mapping species_id -> {
                'usage_pct': float,
                'raw_count': int,
                'moves': {move_id: pct, ...},
                'items': {item_id: pct, ...},
                'abilities': {ability_id: pct, ...},
                'spreads': {spread_str: pct, ...},
                'teammates': {species_id: pct, ...},
                'counters': {species_id: (koed_pct, switched_pct), ...},
            }
        """
        result = {}
        pkmn_data = raw_data.get("data", {})
        total_battles = raw_data.get("info", {}).get("number of battles", 1)

        for species_name, data in pkmn_data.items():
            species_id = self._to_id(species_name)
            raw_count = data.get("Raw count", 0)
            usage_pct = (raw_count / (total_battles * 2)) * 100 if total_battles > 0 else 0

            # Parse sub-dicts: normalize percentages
            moves = self._normalize_weighted(data.get("Moves", {}))
            items = self._normalize_weighted(data.get("Items", {}))
            abilities = self._normalize_weighted(data.get("Abilities", {}))
            spreads = self._normalize_spreads(data.get("Spreads", {}))
            teammates = self._normalize_weighted(data.get("Teammates", {}))

            # Counters have a special format: [koed_pct, switched_pct, ...]
            counters = {}
            for counter_name, counter_data in data.get("Checks and Counters", {}).items():
                if isinstance(counter_data, list) and len(counter_data) >= 2:
                    counters[self._to_id(counter_name)] = (
                        counter_data[0],  # KOed percentage
                        counter_data[1],  # switched out percentage
                    )

            result[species_id] = {
                "usage_pct": usage_pct,
                "raw_count": raw_count,
                "moves": moves,
                "items": items,
                "abilities": abilities,
                "spreads": spreads,
                "teammates": teammates,
                "counters": counters,
            }

        return result

    def get_top_pokemon(
        self, usage_data: dict[str, dict], top_n: int = 50
    ) -> list[dict[str, Any]]:
        """Get the top N Pokemon by usage with their most common sets."""
        sorted_pokemon = sorted(
            usage_data.items(),
            key=lambda x: x[1]["usage_pct"],
            reverse=True,
        )[:top_n]

        result = []
        for species_id, data in sorted_pokemon:
            top_moves = sorted(data["moves"].items(), key=lambda x: x[1], reverse=True)[:8]
            top_items = sorted(data["items"].items(), key=lambda x: x[1], reverse=True)[:4]
            top_abilities = sorted(data["abilities"].items(), key=lambda x: x[1], reverse=True)[:3]
            top_spreads = sorted(data["spreads"].items(), key=lambda x: x[1], reverse=True)[:5]
            top_teammates = sorted(data["teammates"].items(), key=lambda x: x[1], reverse=True)[:10]

            result.append({
                "species": species_id,
                "usage_pct": data["usage_pct"],
                "moves": dict(top_moves),
                "items": dict(top_items),
                "abilities": dict(top_abilities),
                "spreads": dict(top_spreads),
                "teammates": dict(top_teammates),
                "counters": data.get("counters", {}),
            })

        return result

    @staticmethod
    def _normalize_weighted(d: dict) -> dict[str, float]:
        """Normalize a weighted dict so values sum to ~100."""
        total = sum(d.values()) if d else 1
        if total == 0:
            total = 1
        return {
            StatsScraper._to_id(k): (v / total) * 100
            for k, v in d.items()
            if k and k.strip()
        }

    @staticmethod
    def _normalize_spreads(d: dict) -> dict[str, float]:
        """Normalize spread data, preserving Nature:EV format."""
        total = sum(d.values()) if d else 1
        if total == 0:
            total = 1
        return {
            k.strip(): (v / total) * 100
            for k, v in d.items()
            if k and k.strip()
        }

    @staticmethod
    def _to_id(name: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]", "", name.lower())

    @staticmethod
    def _recent_months(n: int) -> list[str]:
        """Generate the last N month strings in YYYY-MM format."""
        months = []
        now = datetime.now()
        # Stats lag by ~1 month, start from 2 months ago
        dt = now.replace(day=1) - timedelta(days=1)
        dt = dt.replace(day=1) - timedelta(days=1)
        for _ in range(n):
            months.append(dt.strftime("%Y-%m"))
            dt = dt.replace(day=1) - timedelta(days=1)
        return months
