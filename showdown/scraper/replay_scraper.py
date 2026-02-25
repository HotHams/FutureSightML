"""Async scraper for Pokemon Showdown replay data."""

import asyncio
import logging
import time
from typing import Any

import aiohttp

from ..data.database import Database
from .replay_parser import ReplayParser

log = logging.getLogger("showdown.scraper.replay_scraper")


class ReplayScraper:
    """Scrape battle replays from Pokemon Showdown's public replay server."""

    def __init__(
        self,
        db: Database,
        base_url: str = "https://replay.pokemonshowdown.com",
        max_concurrent: int = 10,
        request_delay_ms: int = 100,
        min_rating: int = 1500,
    ):
        self.db = db
        self.base_url = base_url.rstrip("/")
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay_ms / 1000.0
        self.min_rating = min_rating
        self.parser = ReplayParser()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._session: aiohttp.ClientSession | None = None
        self._stats = {"fetched": 0, "parsed": 0, "skipped": 0, "errors": 0}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"Accept": "application/json"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def scrape_format(
        self,
        format_id: str,
        max_pages: int = 200,
        progress_callback=None,
    ) -> dict[str, int]:
        """Scrape all available replays for a given format.

        Uses two-phase pagination:
        1. Page-based (page=1..100) — Showdown caps page param at 100.
        2. Cursor-based (before=<timestamp>) — unlimited depth into the past.

        The API forbids combining 'page' and 'before' in the same request.

        Returns statistics dict with counts of fetched/parsed/skipped/errors.
        """
        self._stats = {"fetched": 0, "parsed": 0, "skipped": 0, "errors": 0}
        log.info("Starting scrape for format: %s (max %d pages)", format_id, max_pages)

        total_pages = 0
        consecutive_empty = 0
        oldest_ts_seen = None  # Track the oldest upload timestamp across all results

        # --- Phase 1: page-based pagination (pages 1-100) ---
        page = 1
        while page <= 100 and total_pages < max_pages and consecutive_empty < 5:
            replay_entries = await self._fetch_replay_list(format_id, page=page)

            if not replay_entries:
                consecutive_empty += 1
                page += 1
                total_pages += 1
                continue

            consecutive_empty = 0
            total_pages += 1

            # Track oldest timestamp for transition to cursor pagination
            for entry in replay_entries:
                ts = entry.get("uploadtime")
                if ts and (oldest_ts_seen is None or ts < oldest_ts_seen):
                    oldest_ts_seen = ts

            await self._process_page(replay_entries, format_id, total_pages, progress_callback)
            page += 1

        # --- Phase 2: cursor-based deep pagination (before=timestamp) ---
        if oldest_ts_seen and total_pages < max_pages and consecutive_empty < 5:
            log.info(
                "Switching to cursor pagination (before=%d) after %d pages",
                oldest_ts_seen, total_pages,
            )
            before_ts = oldest_ts_seen
            consecutive_empty = 0

            while total_pages < max_pages and consecutive_empty < 5:
                replay_entries = await self._fetch_replay_list(
                    format_id, before=before_ts
                )

                if not replay_entries:
                    consecutive_empty += 1
                    total_pages += 1
                    continue

                consecutive_empty = 0
                total_pages += 1

                # Advance cursor to the oldest timestamp in this batch
                batch_oldest = None
                for entry in replay_entries:
                    ts = entry.get("uploadtime")
                    if ts and (batch_oldest is None or ts < batch_oldest):
                        batch_oldest = ts

                if batch_oldest is None or batch_oldest >= before_ts:
                    # No progress — stop to avoid infinite loop
                    log.warning("Cursor pagination stalled at before=%d", before_ts)
                    break
                before_ts = batch_oldest

                await self._process_page(replay_entries, format_id, total_pages, progress_callback)

        log.info("Scrape complete for %s: %s", format_id, self._stats)
        return self._stats

    async def _process_page(
        self,
        replay_entries: list[dict],
        format_id: str,
        page_num: int,
        progress_callback=None,
    ) -> None:
        """Process a page of replay entries: filter duplicates, fetch, parse, store."""
        replay_ids = [r["id"] for r in replay_entries]

        # Filter out already-scraped replays
        new_ids = []
        for rid in replay_ids:
            if not await self.db.replay_exists(rid):
                new_ids.append(rid)
            else:
                self._stats["skipped"] += 1

        if new_ids:
            tasks = [self._process_replay(rid, format_id) for rid in new_ids]
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.db.commit()

        if progress_callback:
            progress_callback(page_num, self._stats.copy())

        log.info(
            "Page %d: %d replays (%d new) | Total: %d parsed, %d skipped, %d errors",
            page_num, len(replay_ids), len(new_ids),
            self._stats["parsed"], self._stats["skipped"], self._stats["errors"],
        )

    async def _fetch_replay_list(
        self,
        format_id: str,
        page: int | None = None,
        before: int | None = None,
    ) -> list[dict]:
        """Fetch a page of replay entries for a format.

        Uses either page-based OR cursor-based pagination (never both — the
        Showdown API rejects requests with both 'page' and 'before' set).

        Returns list of dicts with 'id' and 'uploadtime' keys.
        """
        url = f"{self.base_url}/search.json"
        params: dict[str, Any] = {"format": format_id}
        if before is not None:
            params["before"] = before
        elif page is not None:
            params["page"] = page

        try:
            session = await self._get_session()
            async with self._semaphore:
                await asyncio.sleep(self.request_delay)
                async with session.get(url, params=params) as resp:
                    if resp.status == 404:
                        return []
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
        except Exception as e:
            log.error("Failed to fetch replay list (page=%s, before=%s) for %s: %s", page, before, format_id, e)
            return []

        if not isinstance(data, list):
            return []

        return [r for r in data if isinstance(r, dict) and "id" in r]

    async def _process_replay(self, replay_id: str, format_id: str) -> None:
        """Fetch, parse, and store a single replay."""
        try:
            replay_data = await self._fetch_replay(replay_id)
            if replay_data is None:
                self._stats["errors"] += 1
                return

            self._stats["fetched"] += 1

            # Parse the battle log first — it contains both players' ratings
            log_text = replay_data.get("log", "")
            if not log_text:
                self._stats["errors"] += 1
                return

            battle = self.parser.parse(log_text, format_id)

            # Use log-parsed ratings (most reliable: both players' pre-battle Elo)
            rating1 = battle.rating1
            rating2 = battle.rating2

            # Fall back to JSON metadata if log parsing didn't find ratings
            if rating1 is None:
                if isinstance(replay_data.get("p1rating"), dict):
                    rating1 = replay_data["p1rating"].get("elo")
                elif "rating" in replay_data:
                    rating1 = replay_data.get("rating")
            if rating2 is None:
                if isinstance(replay_data.get("p2rating"), dict):
                    rating2 = replay_data["p2rating"].get("elo")

            # Filter by minimum rating
            if self.min_rating > 0:
                r1 = rating1 or 0
                r2 = rating2 or 0
                if r1 < self.min_rating and r2 < self.min_rating:
                    self._stats["skipped"] += 1
                    return

            # Supplement player names from JSON metadata if parser missed them
            if not battle.player1 and replay_data.get("p1"):
                battle.player1 = replay_data["p1"]
            if not battle.player2 and replay_data.get("p2"):
                battle.player2 = replay_data["p2"]

            # Resolve winner from JSON if parser couldn't
            if battle.winner == 0 and hasattr(battle, "_winner_name"):
                wn = battle._winner_name
                if wn and wn == battle.player1:
                    battle.winner = 1
                elif wn and wn == battle.player2:
                    battle.winner = 2

            if not battle.is_valid():
                self._stats["errors"] += 1
                return

            # Store in database
            upload_time = replay_data.get("uploadtime")

            await self.db.insert_replay(
                replay_id=replay_id,
                format_id=format_id,
                player1=battle.player1,
                player2=battle.player2,
                winner=battle.winner,
                rating1=int(rating1) if rating1 else None,
                rating2=int(rating2) if rating2 else None,
                turns=battle.turns,
                upload_time=upload_time,
            )

            team1_data = [p.to_dict() for p in battle.team1]
            team2_data = [p.to_dict() for p in battle.team2]

            await self.db.insert_team(replay_id, 1, team1_data)
            await self.db.insert_team(replay_id, 2, team2_data)

            self._stats["parsed"] += 1

        except Exception as e:
            log.debug("Error processing replay %s: %s", replay_id, e)
            self._stats["errors"] += 1

    async def _fetch_replay(self, replay_id: str) -> dict[str, Any] | None:
        """Fetch a single replay's JSON data."""
        url = f"{self.base_url}/{replay_id}.json"

        try:
            session = await self._get_session()
            async with self._semaphore:
                await asyncio.sleep(self.request_delay)
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return None
                    return await resp.json(content_type=None)
        except Exception as e:
            log.debug("Failed to fetch replay %s: %s", replay_id, e)
            return None
