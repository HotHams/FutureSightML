"""Shared application state: loaded models, data, config."""

import json
import logging
from pathlib import Path

import torch

from ..config import load_config
from ..data.database import Database
from ..data.pokemon_data import PokemonDataLoader
from ..data.features import FeatureExtractor
from ..models.win_predictor import WinPredictor
from ..models.xgb_predictor import XGBPredictor
from ..models.ensemble import EnsemblePredictor
from ..models.trainer import Trainer
from ..teambuilder.constraints import FormatConstraints
from ..teambuilder.evaluator import TeamEvaluator
from ..teambuilder.meta_analysis import MetaAnalyzer

log = logging.getLogger("showdown.api.state")


class AppState:
    """Holds all loaded models and data for the API."""

    def __init__(self):
        self.cfg = None
        self.db: Database | None = None
        self.pkmn_data: PokemonDataLoader | None = None
        self.formats: dict[str, FormatState] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    async def initialize(self, config_path: str | None = None):
        """Load config, connect DB, load Pokemon data and all format models."""
        self.cfg = load_config(config_path)
        db_path = self.cfg.get("database", {}).get("path", "data/showdown.db")
        self.db = Database(db_path)
        await self.db.connect()

        self.pkmn_data = PokemonDataLoader()
        await self.pkmn_data.load()
        log.info("Pokemon data loaded: %d species", len(self.pkmn_data.pokedex))

        # Load models for each format that has checkpoints
        checkpoint_dir = self.cfg.get("training", {}).get("checkpoint_dir", "data/checkpoints")
        all_formats = []
        for fmt_list in self.cfg.get("formats", {}).values():
            all_formats.extend(fmt_list)

        for fmt in all_formats:
            vocab_path = Path(checkpoint_dir) / f"vocab_{fmt}.json"
            if vocab_path.exists():
                try:
                    fs = await self._load_format(fmt, checkpoint_dir)
                    self.formats[fmt] = fs
                    log.info("Loaded format: %s", fmt)
                except Exception as e:
                    log.warning("Failed to load format %s: %s", fmt, e)

        log.info("AppState initialized with %d formats", len(self.formats))

    async def _load_format(self, fmt: str, checkpoint_dir: str) -> "FormatState":
        model_cfg = self.cfg.get("model", {})
        vocab_path = Path(checkpoint_dir) / f"vocab_{fmt}.json"

        with open(vocab_path) as f:
            vocab = json.load(f)

        fe = FeatureExtractor(pokemon_data=self.pkmn_data)
        fe._species_idx = vocab["species"]
        fe._move_idx = vocab["moves"]
        fe._item_idx = vocab["items"]
        fe._ability_idx = vocab["abilities"]

        trainer = Trainer(checkpoint_dir=checkpoint_dir, device=self._device)
        vocab_sizes = fe.vocab_sizes

        # Neural model
        neural_model = None
        try:
            neural_model = WinPredictor(
                num_species=vocab_sizes["species"],
                num_moves=vocab_sizes["moves"],
                num_items=vocab_sizes["items"],
                num_abilities=vocab_sizes["abilities"],
                species_dim=model_cfg.get("pokemon_embed_dim", 64),
                move_dim=model_cfg.get("move_embed_dim", 32),
                item_dim=model_cfg.get("item_embed_dim", 32),
                ability_dim=model_cfg.get("ability_embed_dim", 32),
                pokemon_dim=model_cfg.get("pokemon_hidden_dim", 128),
                team_dim=model_cfg.get("team_hidden_dim", 256),
                attention_heads=model_cfg.get("attention_heads", 4),
                dropout=model_cfg.get("dropout", 0.25),
                continuous_dim=model_cfg.get("continuous_dim", 64),
                rating_dim=model_cfg.get("rating_dim", 6),
            )
            if trainer.load_neural(neural_model, fmt) is None:
                neural_model = None
            else:
                neural_model.eval()
        except Exception as e:
            log.warning("Failed to load neural model for %s: %s", fmt, e)
            neural_model = None

        # XGBoost model
        xgb_model = None
        try:
            xgb_model = XGBPredictor()
            if not trainer.load_xgboost(xgb_model, fmt):
                xgb_model = None
        except Exception:
            xgb_model = None

        ensemble = EnsemblePredictor(
            neural_model=neural_model,
            xgb_model=xgb_model,
            feature_extractor=fe,
            device=self._device,
        )

        # Load calibrated ensemble weights if available
        weights_path = Path(checkpoint_dir) / f"ensemble_{fmt}_weights.json"
        ensemble.load_weights(weights_path)

        # Meta analyzer
        meta_analyzer = MetaAnalyzer(self.db)
        meta_teams = await meta_analyzer.build_meta_teams(fmt, n_teams=50)

        # Pre-compute meta team XGBoost features for fast genetic algo evaluation
        ensemble.precompute_meta_features(meta_teams)

        constraints = FormatConstraints(fmt, pokemon_data=self.pkmn_data)

        # Build pokemon pool (try usage stats first, fall back to replays)
        pokemon_pool = []
        year_month = await self.db.get_latest_usage_month(fmt)
        if year_month:
            for rating in [1825, 1760, 1630, 1500, 0]:
                raw_usage = await self.db.get_usage_stats(fmt, year_month, rating)
                if raw_usage:
                    from ..scraper.stats_scraper import StatsScraper
                    usage_parsed = StatsScraper(self.db).parse_usage_data(raw_usage)
                    pokemon_pool = meta_analyzer.build_full_pokemon_pool(
                        usage_parsed, top_n=80, sets_per_pokemon=4
                    )
                    break

        if not pokemon_pool:
            log.info("No usage stats for %s, building pool from replays...", fmt)
            pokemon_pool = await meta_analyzer.build_pool_from_replays(
                fmt, top_n=80, sets_per_pokemon=4
            )

        return FormatState(
            format_id=fmt,
            feature_extractor=fe,
            ensemble=ensemble,
            meta_teams=meta_teams,
            meta_analyzer=meta_analyzer,
            constraints=constraints,
            pokemon_pool=pokemon_pool,
            vocab=vocab,
        )

    async def shutdown(self):
        if self.db:
            await self.db.close()


class FormatState:
    """State for a single battle format."""

    def __init__(
        self,
        format_id: str,
        feature_extractor: FeatureExtractor,
        ensemble: EnsemblePredictor,
        meta_teams: list[list[dict]],
        meta_analyzer: MetaAnalyzer,
        constraints: FormatConstraints,
        pokemon_pool: list[dict],
        vocab: dict,
    ):
        self.format_id = format_id
        self.fe = feature_extractor
        self.ensemble = ensemble
        self.meta_teams = meta_teams
        self.meta_analyzer = meta_analyzer
        self.constraints = constraints
        self.pokemon_pool = pokemon_pool
        self.vocab = vocab
