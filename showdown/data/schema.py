"""Database schema definitions."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS replays (
    id              TEXT PRIMARY KEY,
    format          TEXT NOT NULL,
    player1         TEXT,
    player2         TEXT,
    winner          INTEGER,  -- 1 or 2 (which player won), 0 for tie
    rating1         INTEGER,
    rating2         INTEGER,
    turns           INTEGER,
    upload_time     INTEGER,
    scraped_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_replays_format ON replays(format);
CREATE INDEX IF NOT EXISTS idx_replays_rating ON replays(rating1, rating2);

CREATE TABLE IF NOT EXISTS battle_teams (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    replay_id       TEXT NOT NULL REFERENCES replays(id),
    player_num      INTEGER NOT NULL,  -- 1 or 2
    team_json       TEXT NOT NULL,      -- JSON array of pokemon sets
    UNIQUE(replay_id, player_num)
);

CREATE INDEX IF NOT EXISTS idx_teams_replay ON battle_teams(replay_id);

CREATE TABLE IF NOT EXISTS pokemon_sets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id         INTEGER NOT NULL REFERENCES battle_teams(id),
    slot            INTEGER NOT NULL,
    species         TEXT NOT NULL,
    ability         TEXT,
    item            TEXT,
    move1           TEXT,
    move2           TEXT,
    move3           TEXT,
    move4           TEXT,
    tera_type       TEXT,
    level           INTEGER DEFAULT 100
);

CREATE INDEX IF NOT EXISTS idx_sets_species ON pokemon_sets(species);
CREATE INDEX IF NOT EXISTS idx_sets_team ON pokemon_sets(team_id);

CREATE TABLE IF NOT EXISTS usage_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    format          TEXT NOT NULL,
    year_month      TEXT NOT NULL,  -- e.g. '2024-01'
    rating_threshold INTEGER NOT NULL,
    total_battles   INTEGER,
    data_json       TEXT NOT NULL,  -- Full chaos JSON
    scraped_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(format, year_month, rating_threshold)
);

CREATE INDEX IF NOT EXISTS idx_usage_format ON usage_stats(format, year_month);

CREATE TABLE IF NOT EXISTS model_checkpoints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    format          TEXT NOT NULL,
    model_type      TEXT NOT NULL,  -- 'neural', 'xgboost', 'ensemble'
    file_path       TEXT NOT NULL,
    accuracy        REAL,
    val_loss        REAL,
    win_pred_auc    REAL,
    metadata_json   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meta_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    format          TEXT NOT NULL,
    year_month      TEXT NOT NULL,
    top_pokemon     TEXT NOT NULL,   -- JSON: [{species, usage_pct, common_sets: [...]}]
    team_archetypes TEXT,            -- JSON: identified team archetypes
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(format, year_month)
);
"""
