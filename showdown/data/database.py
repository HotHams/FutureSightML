"""Async SQLite database layer."""

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite

from .schema import SCHEMA_SQL

log = logging.getLogger("showdown.data.database")


class Database:
    """Async wrapper around SQLite for all persistent storage."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(SCHEMA_SQL)
        await self._conn.commit()
        log.info("Database connected: %s", self.db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    # ------------------------------------------------------------------
    # Replay operations
    # ------------------------------------------------------------------

    async def replay_exists(self, replay_id: str) -> bool:
        cursor = await self.conn.execute(
            "SELECT 1 FROM replays WHERE id = ?", (replay_id,)
        )
        return (await cursor.fetchone()) is not None

    async def insert_replay(
        self,
        replay_id: str,
        format_id: str,
        player1: str,
        player2: str,
        winner: int,
        rating1: int | None,
        rating2: int | None,
        turns: int,
        upload_time: int | None,
    ) -> None:
        await self.conn.execute(
            """INSERT OR IGNORE INTO replays
               (id, format, player1, player2, winner, rating1, rating2, turns, upload_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (replay_id, format_id, player1, player2, winner,
             rating1, rating2, turns, upload_time),
        )

    async def insert_team(
        self,
        replay_id: str,
        player_num: int,
        team_data: list[dict],
    ) -> int:
        team_json = json.dumps(team_data, separators=(",", ":"))
        cursor = await self.conn.execute(
            """INSERT OR IGNORE INTO battle_teams (replay_id, player_num, team_json)
               VALUES (?, ?, ?)""",
            (replay_id, player_num, team_json),
        )
        team_id = cursor.lastrowid

        for slot, pkmn in enumerate(team_data):
            moves = pkmn.get("moves", [])
            await self.conn.execute(
                """INSERT OR IGNORE INTO pokemon_sets
                   (team_id, slot, species, ability, item,
                    move1, move2, move3, move4, tera_type, level)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    team_id, slot,
                    pkmn.get("species", ""),
                    pkmn.get("ability"),
                    pkmn.get("item"),
                    moves[0] if len(moves) > 0 else None,
                    moves[1] if len(moves) > 1 else None,
                    moves[2] if len(moves) > 2 else None,
                    moves[3] if len(moves) > 3 else None,
                    pkmn.get("tera_type"),
                    pkmn.get("level", 100),
                ),
            )
        return team_id

    async def commit(self) -> None:
        await self.conn.commit()

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    async def get_replay_count(self, format_id: str | None = None) -> int:
        if format_id:
            cur = await self.conn.execute(
                "SELECT COUNT(*) FROM replays WHERE format = ?", (format_id,)
            )
        else:
            cur = await self.conn.execute("SELECT COUNT(*) FROM replays")
        row = await cur.fetchone()
        return row[0] if row else 0

    async def get_training_battles(
        self,
        format_id: str,
        min_rating: int = 0,
        limit: int | None = None,
        max_age_days: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch battles with both teams for training. Returns list of dicts."""
        query = """
            SELECT r.id, r.format, r.winner, r.rating1, r.rating2, r.turns,
                   t1.team_json AS team1_json, t2.team_json AS team2_json
            FROM replays r
            JOIN battle_teams t1 ON t1.replay_id = r.id AND t1.player_num = 1
            JOIN battle_teams t2 ON t2.replay_id = r.id AND t2.player_num = 2
            WHERE r.format = ?
              AND r.winner IN (1, 2)
              AND (COALESCE(r.rating1, 0) >= ? OR COALESCE(r.rating2, 0) >= ?)
        """
        params: list[Any] = [format_id, min_rating, min_rating]
        if max_age_days is not None:
            import time
            cutoff = int(time.time()) - max_age_days * 86400
            query += " AND r.upload_time >= ?"
            params.append(cutoff)
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = await self.conn.execute(query, params)
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "replay_id": row["id"],
                "format": row["format"],
                "winner": row["winner"],
                "rating1": row["rating1"],
                "rating2": row["rating2"],
                "turns": row["turns"],
                "team1": json.loads(row["team1_json"]),
                "team2": json.loads(row["team2_json"]),
            })
        return results

    # ------------------------------------------------------------------
    # Usage stats operations
    # ------------------------------------------------------------------

    async def insert_usage_stats(
        self,
        format_id: str,
        year_month: str,
        rating_threshold: int,
        total_battles: int,
        data_json: str,
    ) -> None:
        await self.conn.execute(
            """INSERT OR REPLACE INTO usage_stats
               (format, year_month, rating_threshold, total_battles, data_json)
               VALUES (?, ?, ?, ?, ?)""",
            (format_id, year_month, rating_threshold, total_battles, data_json),
        )

    async def get_usage_stats(
        self,
        format_id: str,
        year_month: str,
        rating_threshold: int = 1825,
    ) -> dict | None:
        cursor = await self.conn.execute(
            """SELECT data_json FROM usage_stats
               WHERE format = ? AND year_month = ? AND rating_threshold = ?""",
            (format_id, year_month, rating_threshold),
        )
        row = await cursor.fetchone()
        return json.loads(row["data_json"]) if row else None

    async def get_latest_usage_month(self, format_id: str) -> str | None:
        cursor = await self.conn.execute(
            """SELECT year_month FROM usage_stats
               WHERE format = ? ORDER BY year_month DESC LIMIT 1""",
            (format_id,),
        )
        row = await cursor.fetchone()
        return row["year_month"] if row else None

    # ------------------------------------------------------------------
    # Checkpoint operations
    # ------------------------------------------------------------------

    async def save_checkpoint_meta(
        self,
        format_id: str,
        model_type: str,
        file_path: str,
        accuracy: float | None = None,
        val_loss: float | None = None,
        auc: float | None = None,
        metadata: dict | None = None,
    ) -> int:
        cursor = await self.conn.execute(
            """INSERT INTO model_checkpoints
               (format, model_type, file_path, accuracy, val_loss, win_pred_auc, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (format_id, model_type, file_path, accuracy, val_loss, auc,
             json.dumps(metadata) if metadata else None),
        )
        await self.conn.commit()
        return cursor.lastrowid

    async def get_best_checkpoint(
        self, format_id: str, model_type: str
    ) -> dict | None:
        cursor = await self.conn.execute(
            """SELECT * FROM model_checkpoints
               WHERE format = ? AND model_type = ?
               ORDER BY win_pred_auc DESC NULLS LAST, accuracy DESC NULLS LAST
               LIMIT 1""",
            (format_id, model_type),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    # ------------------------------------------------------------------
    # Meta snapshot operations
    # ------------------------------------------------------------------

    async def save_meta_snapshot(
        self,
        format_id: str,
        year_month: str,
        top_pokemon: list[dict],
        team_archetypes: list[dict] | None = None,
    ) -> None:
        await self.conn.execute(
            """INSERT OR REPLACE INTO meta_snapshots
               (format, year_month, top_pokemon, team_archetypes)
               VALUES (?, ?, ?, ?)""",
            (format_id, year_month, json.dumps(top_pokemon),
             json.dumps(team_archetypes) if team_archetypes else None),
        )
        await self.conn.commit()

    async def get_meta_snapshot(
        self, format_id: str, year_month: str | None = None
    ) -> dict | None:
        if year_month:
            cursor = await self.conn.execute(
                "SELECT * FROM meta_snapshots WHERE format = ? AND year_month = ?",
                (format_id, year_month),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM meta_snapshots WHERE format = ? ORDER BY year_month DESC LIMIT 1",
                (format_id,),
            )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "format": row["format"],
            "year_month": row["year_month"],
            "top_pokemon": json.loads(row["top_pokemon"]),
            "team_archetypes": json.loads(row["team_archetypes"]) if row["team_archetypes"] else None,
        }
