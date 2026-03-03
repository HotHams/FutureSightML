"""FastAPI server for FutureSightML."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .state import AppState
from ..teambuilder.evaluator import TeamEvaluator
from ..teambuilder.genetic import GeneticTeamBuilder
from ..teambuilder.analysis import TeamAnalyzer
from ..teambuilder.spread_inference import apply_spreads
from ..simulator import BattleSimulator, MonteCarloSimulator
from ..utils.constants import extract_gen
from ..utils.logging_config import setup_logging
from ..config import get_data_root

import sys as _sys
if getattr(_sys, '_MEIPASS', None):
    # PyInstaller bundle: _MEIPASS is read-only, write logs next to the executable
    _log_dir = Path(_sys.executable).parent / "logs"
else:
    _log_dir = get_data_root() / "data" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
_log_file = str(_log_dir / "server.log")
log = setup_logging("INFO", log_file=_log_file)

app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_state.initialize()
    yield
    await app_state.shutdown()


app = FastAPI(
    title="FutureSightML",
    description="Pokemon Showdown team builder powered by ML",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*", "file://"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the GUI
from ..config import resolve_data_path as _resolve
_gui_dir = _resolve("gui/static")
if not _gui_dir.exists():
    _gui_dir = Path(__file__).resolve().parent.parent.parent / "gui" / "static"
if _gui_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_gui_dir)), name="static")


@app.get("/")
async def root():
    """Serve the main GUI page."""
    from fastapi.responses import FileResponse
    index = _gui_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "FutureSightML API", "docs": "/docs"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PokemonSet(BaseModel):
    model_config = ConfigDict(extra="ignore")
    species: str
    ability: str | None = ""
    item: str | None = ""
    moves: list[str] = Field(default_factory=list)
    tera_type: str | None = ""
    evs: dict[str, int] | None = Field(default_factory=dict)
    ivs: dict[str, int] | None = Field(default_factory=dict)
    nature: str | None = ""
    level: int | None = 100


class TeamRequest(BaseModel):
    team: list[PokemonSet]


class GenerateRequest(BaseModel):
    format_id: str = "gen9ou"
    n_results: int = Field(5, ge=1, le=50)
    population: int = Field(100, ge=10, le=500)
    generations: int = Field(50, ge=5, le=200)
    mutation_rate: float = Field(0.15, ge=0.0, le=1.0)


class EvaluateRequest(BaseModel):
    format_id: str = "gen9ou"
    team: list[PokemonSet]


class BattleRequest(BaseModel):
    format_id: str = "gen9ou"
    team1: list[PokemonSet]
    team2: list[PokemonSet]


class SimulateRequest(BaseModel):
    team1: list[PokemonSet]
    team2: list[PokemonSet]
    n_simulations: int = 100


class ImportRequest(BaseModel):
    format_id: str = "gen9ou"
    paste: str


class HeadToHeadRequest(BaseModel):
    format_id: str = "gen9ou"
    team1: list[PokemonSet]
    team2: list[PokemonSet]


class DamageCalcRequest(BaseModel):
    format_id: str = "gen9ou"
    attacker: PokemonSet
    defender: PokemonSet
    move: str


class CounterRequest(BaseModel):
    format_id: str = "gen9ou"
    pokemon: PokemonSet
    n: int = 10


class SlotSuggestionRequest(BaseModel):
    format_id: str = "gen9ou"
    team: list[PokemonSet]
    slot_idx: int


class CompareRequest(BaseModel):
    format_id: str = "gen9ou"
    team1: list[PokemonSet]
    team2: list[PokemonSet]


# ---------------------------------------------------------------------------
# Routes: Formats
# ---------------------------------------------------------------------------

@app.get("/api/formats")
async def list_formats():
    """List all available formats with model status."""
    formats = []
    all_fmts = []
    for cat, fmt_list in app_state.cfg.get("formats", {}).items():
        for fmt in fmt_list:
            has_model = fmt in app_state.formats
            fs = app_state.formats.get(fmt)
            formats.append({
                "id": fmt,
                "category": cat,
                "has_model": has_model,
                "meta_teams": len(fs.meta_teams) if fs else 0,
                "pool_size": len(fs.pokemon_pool) if fs else 0,
            })
    return {"formats": formats}


# ---------------------------------------------------------------------------
# Routes: Team Generation
# ---------------------------------------------------------------------------

@app.post("/api/team/generate")
async def generate_team(req: GenerateRequest):
    """Generate optimized teams using the genetic algorithm."""
    fmt = req.format_id
    if fmt not in app_state.formats:
        raise HTTPException(404, f"Format '{fmt}' not loaded. Available: {list(app_state.formats.keys())}")

    fs = app_state.formats[fmt]
    if not fs.meta_teams:
        raise HTTPException(400, "No meta teams available. Scrape usage stats first.")
    if len(fs.pokemon_pool) < 6:
        raise HTTPException(
            422,
            "Not enough Pokemon data to generate teams for this format. "
            f"Only {len(fs.pokemon_pool)} sets available (need at least 6). "
            "Try a more popular format.",
        )

    gen = extract_gen(fmt)

    evaluator = TeamEvaluator(
        predictor=fs.ensemble,
        meta_teams=fs.meta_teams,
        fast_mode=True,
    )
    builder = GeneticTeamBuilder(
        evaluator=evaluator,
        constraints=fs.constraints,
        pokemon_pool=fs.pokemon_pool,
        population_size=req.population,
        generations=req.generations,
        mutation_rate=req.mutation_rate,
    )

    # Run in thread pool to not block event loop
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, builder.build, req.n_results)

    # Build a standard evaluator (non-fast) to compute true predicted win rates
    # that match what /api/team/evaluate returns.
    std_evaluator = TeamEvaluator(
        predictor=fs.ensemble,
        meta_teams=fs.meta_teams,
    )

    teams = []
    for r in results:
        # Apply inferred EV spreads and natures (gen-aware)
        enriched_team = apply_spreads(r["team"], app_state.pkmn_data, gen=gen)
        team_data = []
        for p in enriched_team:
            pkmn_out = {
                "species": p.get("species", ""),
                "moves": p.get("moves", []),
            }
            # Gen 3+: abilities exist
            if gen >= 3:
                pkmn_out["ability"] = p.get("ability", "")
            # Gen 2+: items exist
            if gen >= 2:
                pkmn_out["item"] = p.get("item", "")
            # Gen 3+: natures and EVs exist
            if gen >= 3:
                pkmn_out["nature"] = p.get("nature", "")
                pkmn_out["evs"] = p.get("evs", {})
                pkmn_out["ivs"] = p.get("ivs", {})
            # Gen 9 only: tera types
            if gen >= 9:
                pkmn_out["tera_type"] = p.get("tera_type", "")
            team_data.append(pkmn_out)

        # Compute true predicted win rate (same as /api/team/evaluate)
        detailed = std_evaluator.evaluate_detailed(r["team"])
        predicted_wr = round(detailed["overall_winrate"] * 100, 2)

        teams.append({
            "pokemon": team_data,
            "predicted_winrate": predicted_wr,
            "generation_found": r["generation"],
        })

    return {"format": fmt, "gen": gen, "teams": teams}


@app.post("/api/team/evaluate")
async def evaluate_team(req: EvaluateRequest):
    """Evaluate a team's predicted win rate against the meta."""
    fmt = req.format_id
    if fmt not in app_state.formats:
        raise HTTPException(404, f"Format '{fmt}' not loaded")

    fs = app_state.formats[fmt]
    if not fs.meta_teams:
        raise HTTPException(400, "No meta teams available")

    team = [p.model_dump() for p in req.team]
    evaluator = TeamEvaluator(predictor=fs.ensemble, meta_teams=fs.meta_teams)
    detailed = evaluator.evaluate_detailed(team)

    return {
        "format": fmt,
        "overall_winrate": round(detailed["overall_winrate"] * 100, 2),
        "matchups_above_50": detailed["matchups_above_50"],
        "total_matchups": detailed["total_matchups"],
        "worst_matchups": [
            {
                "opponent": m["opponent"],
                "win_prob": round(m["win_prob"] * 100, 2),
            }
            for m in detailed["worst_matchups"]
        ],
        "best_matchups": [
            {
                "opponent": m["opponent"],
                "win_prob": round(m["win_prob"] * 100, 2),
            }
            for m in detailed["best_matchups"]
        ],
    }


@app.post("/api/team/predict")
async def predict_battle(req: BattleRequest):
    """Predict win probability for team1 vs team2."""
    fmt = req.format_id
    if fmt not in app_state.formats:
        raise HTTPException(404, f"Format '{fmt}' not loaded")

    fs = app_state.formats[fmt]
    battle = {
        "team1": [p.model_dump() for p in req.team1],
        "team2": [p.model_dump() for p in req.team2],
        "winner": 0,
    }
    pred = fs.ensemble.predict_battle(battle)

    return {
        "team1_win_prob": round(pred["ensemble"] * 100, 2),
        "team2_win_prob": round((1 - pred["ensemble"]) * 100, 2),
        "neural_prob": round(pred.get("neural", 0.5) * 100, 2) if "neural" in pred else None,
        "xgboost_prob": round(pred.get("xgboost", 0.5) * 100, 2) if "xgboost" in pred else None,
    }


# ---------------------------------------------------------------------------
# Routes: Battle Simulation
# ---------------------------------------------------------------------------

@app.post("/api/battle/simulate")
async def simulate_battle(req: SimulateRequest):
    """Simulate battles between two teams using the Monte Carlo engine.

    Runs turn-by-turn battle simulations with real Pokemon mechanics
    (type effectiveness, STAB, priority, hazards, status, items, abilities)
    and returns win rate estimates with detailed statistics.
    """
    if not app_state.pkmn_data:
        raise HTTPException(500, "Pokemon data not loaded")

    team1 = [p.model_dump() for p in req.team1]
    team2 = [p.model_dump() for p in req.team2]
    n = min(req.n_simulations, 1000)  # Cap at 1000

    mc = MonteCarloSimulator(app_state.pkmn_data, n_simulations=n)

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, mc.estimate_win_rate, team1, team2, n)

    # Summarize — don't return every single simulation detail
    return {
        "team1_win_rate": round(results["team1_win_rate"] * 100, 2),
        "team2_win_rate": round(results["team2_win_rate"] * 100, 2),
        "tie_rate": round(results["tie_rate"] * 100, 2),
        "avg_turns": round(results["avg_turns"], 1),
        "avg_team1_remaining": round(results["avg_team1_remaining"], 2),
        "avg_team2_remaining": round(results["avg_team2_remaining"], 2),
        "n_simulations": results["n_simulations"],
    }


@app.post("/api/battle/simulate/single")
async def simulate_single_battle(req: BattleRequest):
    """Run a single battle simulation with turn-by-turn details."""
    if not app_state.pkmn_data:
        raise HTTPException(500, "Pokemon data not loaded")

    team1 = [p.model_dump() for p in req.team1]
    team2 = [p.model_dump() for p in req.team2]

    sim = BattleSimulator(app_state.pkmn_data)
    result = sim.simulate(team1, team2)

    return {
        "winner": result["winner"],
        "turns": result["turns"],
        "team1_remaining": result["team1_remaining"],
        "team2_remaining": result["team2_remaining"],
        "team1_hp_pct": round(result["team1_hp_pct"] * 100, 2),
        "team2_hp_pct": round(result["team2_hp_pct"] * 100, 2),
    }


# ---------------------------------------------------------------------------
# Routes: Import/Export
# ---------------------------------------------------------------------------

@app.post("/api/team/import")
async def import_team(req: ImportRequest):
    """Import a team from Pokemon Showdown paste format."""
    team = _parse_showdown_paste(req.paste)
    if not team:
        raise HTTPException(400, "Could not parse team paste")

    # Optionally evaluate if format has models
    evaluation = None
    if req.format_id in app_state.formats:
        fs = app_state.formats[req.format_id]
        if fs.meta_teams:
            evaluator = TeamEvaluator(predictor=fs.ensemble, meta_teams=fs.meta_teams)
            detailed = evaluator.evaluate_detailed([p for p in team])
            evaluation = {
                "overall_winrate": round(detailed["overall_winrate"] * 100, 2),
                "matchups_above_50": detailed["matchups_above_50"],
                "total_matchups": detailed["total_matchups"],
            }

    return {"team": team, "evaluation": evaluation}


@app.post("/api/team/export")
async def export_team(req: TeamRequest):
    """Export a team to Pokemon Showdown paste format."""
    paste = _team_to_showdown_paste(req.team)
    return {"paste": paste}


# ---------------------------------------------------------------------------
# Routes: Pokemon Data
# ---------------------------------------------------------------------------

@app.get("/api/pokemon/search")
async def search_pokemon(q: str = Query(..., min_length=1)):
    """Search Pokemon by name prefix."""
    if not app_state.pkmn_data:
        raise HTTPException(500, "Pokemon data not loaded")

    results = []
    q_lower = q.lower().replace(" ", "").replace("-", "")
    for name, data in app_state.pkmn_data.pokedex.items():
        if q_lower in name:
            types = data.get("types", [])
            stats = data.get("baseStats", {})
            results.append({
                "id": name,
                "name": data.get("name", name),
                "types": types,
                "baseStats": stats,
                "abilities": list((data.get("abilities") or {}).values()),
                "sprite": _sprite_url(name),
            })
            if len(results) >= 20:
                break

    return {"results": results}


@app.get("/api/pokemon/{pokemon_id}")
async def get_pokemon(pokemon_id: str):
    """Get detailed Pokemon data."""
    if not app_state.pkmn_data:
        raise HTTPException(500, "Pokemon data not loaded")

    pid = pokemon_id.lower().replace(" ", "").replace("-", "")
    data = app_state.pkmn_data.pokedex.get(pid)
    if not data:
        raise HTTPException(404, f"Pokemon '{pokemon_id}' not found")

    # Get moves this Pokemon can actually learn (from learnset data)
    learnable_moves = []
    ls_entry = app_state.pkmn_data.learnsets.get(pid, {}).get("learnset", {})
    for move_id in ls_entry:
        move_data = app_state.pkmn_data.moves.get(move_id)
        if move_data:
            learnable_moves.append({
                "id": move_id,
                "name": move_data.get("name", move_id),
                "type": move_data.get("type", ""),
                "category": move_data.get("category", ""),
                "basePower": move_data.get("basePower", 0),
                "accuracy": move_data.get("accuracy", 0),
            })

    return {
        "id": pid,
        "name": data.get("name", pid),
        "types": data.get("types", []),
        "baseStats": data.get("baseStats", {}),
        "abilities": data.get("abilities", {}),
        "moves": learnable_moves,
        "sprite": _sprite_url(pid),
        "sprite_ani": _sprite_url(pid, animated=True),
    }


@app.get("/api/pokemon/{pokemon_id}/usage")
async def get_pokemon_usage(pokemon_id: str, format_id: str = "gen9ou"):
    """Get usage stats for a Pokemon in a format."""
    if format_id not in app_state.formats:
        raise HTTPException(404, f"Format '{format_id}' not loaded")

    fs = app_state.formats[format_id]
    pid = pokemon_id.lower().replace(" ", "").replace("-", "")

    # Find in pokemon pool
    sets = [p for p in fs.pokemon_pool if p.get("species", "").lower().replace(" ", "").replace("-", "") == pid]

    return {
        "pokemon": pid,
        "format": format_id,
        "sets_in_pool": len(sets),
        "sets": sets[:10],
    }


# ---------------------------------------------------------------------------
# Routes: Meta Analysis
# ---------------------------------------------------------------------------

@app.get("/api/meta/{format_id}")
async def get_meta_analysis(format_id: str):
    """Get metagame analysis for a format."""
    if format_id not in app_state.formats:
        raise HTTPException(404, f"Format '{format_id}' not loaded")

    fs = app_state.formats[format_id]

    # Get usage stats summary
    usage_summary = []
    if fs.pokemon_pool:
        species_count = {}
        for p in fs.pokemon_pool:
            sp = p.get("species", "")
            if sp not in species_count:
                species_count[sp] = {"species": sp, "sets": 0, "sprite": _sprite_url(sp)}
            species_count[sp]["sets"] += 1
        usage_summary = sorted(species_count.values(), key=lambda x: -x["sets"])[:50]

    # Meta teams summary
    meta_summary = []
    for team in fs.meta_teams[:10]:
        species_list = [p.get("species", "?") for p in team[:6]]
        meta_summary.append({
            "pokemon": species_list,
            "sprites": [_sprite_url(s) for s in species_list],
        })

    return {
        "format": format_id,
        "top_pokemon": usage_summary,
        "meta_teams": meta_summary,
        "total_meta_teams": len(fs.meta_teams),
        "pool_size": len(fs.pokemon_pool),
    }


@app.get("/api/meta/{format_id}/stats")
async def get_format_stats(format_id: str):
    """Get database stats for a format."""
    if not app_state.db:
        raise HTTPException(500, "Database not connected")

    replay_count = await app_state.db.get_replay_count(format_id)
    has_model = format_id in app_state.formats

    return {
        "format": format_id,
        "replay_count": replay_count,
        "has_model": has_model,
    }


# ---------------------------------------------------------------------------
# Routes: Advanced Analysis (Tier 2-3)
# ---------------------------------------------------------------------------

@app.post("/api/team/analyze/speed")
async def analyze_speed_tiers(req: EvaluateRequest):
    """Analyze speed tiers for each team member."""
    if not app_state.pkmn_data:
        raise HTTPException(500, "Pokemon data not loaded")
    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    team = [p.model_dump() for p in req.team]
    result = analyzer.speed_tier_analysis(team)
    return {"format": req.format_id, "speed_tiers": result}


@app.post("/api/team/analyze/archetype")
async def analyze_archetype(req: EvaluateRequest):
    """Detect the team's playstyle archetype."""
    if not app_state.pkmn_data:
        raise HTTPException(500, "Pokemon data not loaded")
    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    team = [p.model_dump() for p in req.team]
    result = analyzer.detect_archetype(team)
    return {"format": req.format_id, **result}


@app.post("/api/team/analyze/tera")
async def analyze_tera(req: EvaluateRequest):
    """Suggest optimal tera types for each team member."""
    if not app_state.pkmn_data:
        raise HTTPException(500, "Pokemon data not loaded")
    fmt = req.format_id
    meta_teams = []
    if fmt in app_state.formats:
        meta_teams = app_state.formats[fmt].meta_teams

    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    team = [p.model_dump() for p in req.team]
    result = analyzer.optimize_tera_types(team, meta_teams)
    return {"format": fmt, "tera_suggestions": result}


@app.post("/api/team/analyze/threats")
async def analyze_threats(req: EvaluateRequest):
    """Identify biggest threats to this team from the meta."""
    fmt = req.format_id
    if fmt not in app_state.formats:
        raise HTTPException(404, f"Format '{fmt}' not loaded")

    fs = app_state.formats[fmt]
    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    team = [p.model_dump() for p in req.team]
    result = analyzer.analyze_threats(team, fs.meta_teams)
    return {"format": fmt, **result}


@app.post("/api/team/analyze/coverage")
async def analyze_coverage(req: EvaluateRequest):
    """Analyze offensive and defensive type coverage."""
    if not app_state.pkmn_data:
        raise HTTPException(500, "Pokemon data not loaded")
    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    team = [p.model_dump() for p in req.team]
    result = analyzer.coverage_analysis(team)
    return {"format": req.format_id, **result}


@app.post("/api/team/analyze/strategy")
async def analyze_strategy(req: EvaluateRequest):
    """Explain the team's strategy, roles, win conditions, and game plan."""
    if not app_state.pkmn_data:
        raise HTTPException(500, "Pokemon data not loaded")
    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    team = [p.model_dump() for p in req.team]
    result = analyzer.explain_strategy(team)
    return {"format": req.format_id, **result}


@app.post("/api/team/analyze/full")
async def full_analysis(req: EvaluateRequest):
    """Run all analysis tools on a team at once."""
    if not app_state.pkmn_data:
        raise HTTPException(500, "Pokemon data not loaded")
    fmt = req.format_id
    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    team = [p.model_dump() for p in req.team]

    meta_teams = []
    if fmt in app_state.formats:
        meta_teams = app_state.formats[fmt].meta_teams

    # Run all analyses
    speed = analyzer.speed_tier_analysis(team)
    archetype = analyzer.detect_archetype(team)
    tera = analyzer.optimize_tera_types(team, meta_teams)
    coverage = analyzer.coverage_analysis(team)
    strategy = analyzer.explain_strategy(team)

    result = {
        "format": fmt,
        "speed_tiers": speed,
        "archetype": archetype,
        "tera_suggestions": tera,
        "coverage": coverage,
        "strategy": strategy,
    }

    # Add threat analysis if meta teams available
    if meta_teams:
        threats = analyzer.analyze_threats(team, meta_teams)
        result["threats"] = threats

    # Add evaluation if model available
    if fmt in app_state.formats:
        fs = app_state.formats[fmt]
        if fs.meta_teams:
            evaluator = TeamEvaluator(predictor=fs.ensemble, meta_teams=fs.meta_teams)
            detailed = evaluator.evaluate_detailed(team)
            result["evaluation"] = {
                "overall_winrate": round(detailed["overall_winrate"] * 100, 2),
                "matchups_above_50": detailed["matchups_above_50"],
                "total_matchups": detailed["total_matchups"],
            }

    return result


# ---------------------------------------------------------------------------
# Routes: Community Features
# ---------------------------------------------------------------------------

@app.post("/api/team/head-to-head")
async def head_to_head(req: HeadToHeadRequest):
    """Compute pairwise matchup matrix and win probability between two teams."""
    if not app_state.pkmn_data:
        raise HTTPException(500, "Pokemon data not loaded")
    fmt = req.format_id
    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    team1 = [p.model_dump() for p in req.team1]
    team2 = [p.model_dump() for p in req.team2]

    gen = extract_gen(fmt)
    from ..data.features import FeatureExtractor
    fe = FeatureExtractor(app_state.pkmn_data, gen=gen)
    matrix = analyzer.pairwise_matchup_matrix(team1, team2, feature_extractor=fe)

    # Win probability if model available
    win_prob = None
    if fmt in app_state.formats:
        fs = app_state.formats[fmt]
        battle = {"team1": team1, "team2": team2, "winner": 0}
        pred = fs.ensemble.predict_battle(battle)
        win_prob = {
            "team1_win_prob": round(pred["ensemble"] * 100, 2),
            "team2_win_prob": round((1 - pred["ensemble"]) * 100, 2),
        }

    return {
        "format": fmt,
        "matchup_matrix": matrix,
        "win_probability": win_prob,
    }


@app.post("/api/battle/damage")
async def damage_calc(req: DamageCalcRequest):
    """Calculate damage for a specific move from attacker to defender."""
    if not app_state.pkmn_data:
        raise HTTPException(500, "Pokemon data not loaded")

    import re
    gen = extract_gen(req.format_id)
    from ..data.features import FeatureExtractor
    from ..data.damage_calc import estimate_damage_pct as _est_dmg

    fe = FeatureExtractor(app_state.pkmn_data, gen=gen)
    atk = [req.attacker.model_dump()]
    dfn = [req.defender.model_dump()]
    atk_data = fe.precompute_team_data(atk)
    def_data = fe.precompute_team_data(dfn)

    if not atk_data or not def_data:
        raise HTTPException(400, "Could not process Pokemon data")

    atk_pre = atk_data[0]
    def_pre = def_data[0]

    # Find the requested move
    move_id = re.sub(r"[^a-z0-9]", "", req.move.lower())
    move_data = None
    for md in atk_pre.get("moves_data", []):
        md_id = re.sub(r"[^a-z0-9]", "", md.get("name", "").lower())
        if md_id == move_id:
            move_data = md
            break

    if not move_data:
        # Try looking up from pokemon data
        if move_id in app_state.pkmn_data.moves:
            raw = app_state.pkmn_data.moves[move_id]
            move_data = {
                "name": raw.get("name", req.move),
                "basePower": raw.get("basePower", 0),
                "category": raw.get("category", "Status"),
                "type": raw.get("type", "Normal"),
                "flags": raw.get("flags", {}),
                "secondary": raw.get("secondary"),
                "overrideOffensiveStat": None,
                "overrideOffensivePokemon": None,
                "overrideDefensiveStat": None,
            }

    if not move_data or move_data.get("category") == "Status":
        return {
            "damage_pct": 0.0, "min_roll": 0.0, "max_roll": 0.0,
            "is_ohko": False, "is_2hko": False, "move_type": move_data.get("type", "") if move_data else "",
            "effectiveness": "neutral",
        }

    dmg = _est_dmg(atk_pre, def_pre, move_data)
    dmg_pct = round(float(dmg * 100), 1)
    min_roll = round(float(dmg * 0.85 * 100), 1)
    max_roll = round(float(dmg * 100), 1)

    # Type effectiveness
    from ..utils.constants import type_effectiveness_against
    def_types = def_pre.get("types", [])
    move_type = move_data.get("type", "Normal")
    eff = type_effectiveness_against(move_type, def_types) if def_types else 1.0
    eff_label = "neutral"
    if eff >= 4.0:
        eff_label = "double_super"
    elif eff >= 2.0:
        eff_label = "super"
    elif eff == 0:
        eff_label = "immune"
    elif eff <= 0.25:
        eff_label = "double_resist"
    elif eff < 1.0:
        eff_label = "resist"

    return {
        "damage_pct": dmg_pct,
        "min_roll": min_roll,
        "max_roll": max_roll,
        "is_ohko": min_roll >= 100.0,
        "is_2hko": min_roll >= 50.0,
        "move_type": move_type,
        "move_name": move_data.get("name", req.move),
        "effectiveness": eff_label,
        "effectiveness_mult": float(eff),
    }


@app.post("/api/team/counters")
async def find_counters(req: CounterRequest):
    """Find counters and checks for a Pokemon from the format's pool."""
    fmt = req.format_id
    if fmt not in app_state.formats:
        raise HTTPException(404, f"Format '{fmt}' not loaded")

    fs = app_state.formats[fmt]
    gen = extract_gen(fmt)
    from ..data.features import FeatureExtractor
    fe = FeatureExtractor(app_state.pkmn_data, gen=gen)

    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    target = req.pokemon.model_dump()

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, analyzer.find_counters, target, fs.pokemon_pool, req.n, fe,
    )

    return {"format": fmt, "target": target.get("species", ""), **result}


@app.post("/api/team/suggest-slot")
async def suggest_slot(req: SlotSuggestionRequest):
    """Suggest replacement Pokemon for a specific team slot."""
    fmt = req.format_id
    if fmt not in app_state.formats:
        raise HTTPException(404, f"Format '{fmt}' not loaded")

    fs = app_state.formats[fmt]
    if not fs.meta_teams:
        raise HTTPException(400, "No meta teams available for evaluation")

    team = [p.model_dump() for p in req.team]
    evaluator = TeamEvaluator(
        predictor=fs.ensemble,
        meta_teams=fs.meta_teams,
        fast_mode=True,
    )
    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, analyzer.suggest_slot_replacement,
        team, req.slot_idx, fs.pokemon_pool, evaluator, fs.meta_teams,
    )

    return {"format": fmt, "slot_idx": req.slot_idx, "suggestions": results}


@app.post("/api/team/compare")
async def compare_teams(req: CompareRequest):
    """Compare two teams side by side."""
    fmt = req.format_id
    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    team1 = [p.model_dump() for p in req.team1]
    team2 = [p.model_dump() for p in req.team2]

    cov1 = analyzer.coverage_analysis(team1)
    cov2 = analyzer.coverage_analysis(team2)
    speed1 = analyzer.speed_tier_analysis(team1)
    speed2 = analyzer.speed_tier_analysis(team2)
    arch1 = analyzer.detect_archetype(team1)
    arch2 = analyzer.detect_archetype(team2)

    # Win rates if model available
    wr1 = wr2 = None
    if fmt in app_state.formats:
        fs = app_state.formats[fmt]
        if fs.meta_teams:
            evaluator = TeamEvaluator(predictor=fs.ensemble, meta_teams=fs.meta_teams)
            d1 = evaluator.evaluate_detailed(team1)
            d2 = evaluator.evaluate_detailed(team2)
            wr1 = round(d1["overall_winrate"] * 100, 2)
            wr2 = round(d2["overall_winrate"] * 100, 2)

    # Shared / unique Pokemon
    sp1 = {p.get("species", "") for p in team1 if p}
    sp2 = {p.get("species", "") for p in team2 if p}
    shared = list(sp1 & sp2)
    unique1 = list(sp1 - sp2)
    unique2 = list(sp2 - sp1)

    return {
        "format": fmt,
        "team1_wr": wr1,
        "team2_wr": wr2,
        "wr_delta": round(wr1 - wr2, 2) if wr1 is not None and wr2 is not None else None,
        "team1_archetype": arch1.get("archetype", ""),
        "team2_archetype": arch2.get("archetype", ""),
        "team1_coverage": {
            "uncovered": cov1.get("uncovered_types", []),
            "unresisted": cov1.get("unresisted_types", []),
        },
        "team2_coverage": {
            "uncovered": cov2.get("uncovered_types", []),
            "unresisted": cov2.get("unresisted_types", []),
        },
        "shared_pokemon": shared,
        "unique_to_team1": unique1,
        "unique_to_team2": unique2,
        "team1_speed": [{"species": s["species"], "max_speed": s["max_speed"]} for s in speed1],
        "team2_speed": [{"species": s["species"], "max_speed": s["max_speed"]} for s in speed2],
    }


@app.get("/api/meta/{format_id}/pokemon/{species}")
async def get_species_usage(format_id: str, species: str):
    """Get all sets for a specific species in a format's pool."""
    if format_id not in app_state.formats:
        raise HTTPException(404, f"Format '{format_id}' not loaded")

    import re
    fs = app_state.formats[format_id]
    pid = re.sub(r"[^a-z0-9]", "", species.lower())

    sets = [p for p in fs.pokemon_pool
            if re.sub(r"[^a-z0-9]", "", p.get("species", "").lower()) == pid]

    # Aggregate move/item/ability usage
    move_counts = {}
    item_counts = {}
    ability_counts = {}
    for s in sets:
        for m in s.get("moves", []):
            if m:
                move_counts[m] = move_counts.get(m, 0) + 1
        itm = s.get("item", "")
        if itm:
            item_counts[itm] = item_counts.get(itm, 0) + 1
        abl = s.get("ability", "")
        if abl:
            ability_counts[abl] = ability_counts.get(abl, 0) + 1

    total = max(len(sets), 1)
    top_moves = sorted(move_counts.items(), key=lambda x: -x[1])[:10]
    top_items = sorted(item_counts.items(), key=lambda x: -x[1])[:5]
    top_abilities = sorted(ability_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "format": format_id,
        "species": species,
        "total_sets": len(sets),
        "top_moves": [{"name": m, "count": c, "pct": round(c / total * 100, 1)} for m, c in top_moves],
        "top_items": [{"name": i, "count": c, "pct": round(c / total * 100, 1)} for i, c in top_items],
        "top_abilities": [{"name": a, "count": c, "pct": round(c / total * 100, 1)} for a, c in top_abilities],
        "sets": sets[:10],
    }


@app.get("/api/formats/tier-list")
async def formats_tier_list():
    """Get cross-format usage data for a tier list heatmap."""
    import re

    format_data = {}
    pokemon_usage = {}  # species -> {format: pct}

    for fmt_id, fs in app_state.formats.items():
        if not fs.pokemon_pool:
            continue

        species_count = {}
        for p in fs.pokemon_pool:
            sp = p.get("species", "")
            if sp:
                species_count[sp] = species_count.get(sp, 0) + 1

        total = sum(species_count.values())
        if total == 0:
            continue

        top20 = sorted(species_count.items(), key=lambda x: -x[1])[:20]
        format_data[fmt_id] = {
            "total_sets": total,
            "top_pokemon": [s for s, _ in top20],
        }

        for sp, count in top20:
            if sp not in pokemon_usage:
                pokemon_usage[sp] = {}
            pokemon_usage[sp][fmt_id] = round(count / total, 4)

    # Rank by total cross-format presence
    ranked = sorted(
        pokemon_usage.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True,
    )[:30]

    return {
        "formats": list(format_data.keys()),
        "pokemon": [
            {
                "species": sp,
                "sprite": _sprite_url(sp),
                "usage": usage,
                "total_usage": round(sum(usage.values()), 4),
            }
            for sp, usage in ranked
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sprite_id(species: str) -> str:
    """Convert a species name/ID to a Showdown sprite-compatible name.

    Inserts hyphens before known forme suffixes since Showdown sprites
    use hyphenated names (e.g. 'weezing-galar') while our IDs are
    concatenated (e.g. 'weezinggalar').
    """
    import re
    name = re.sub(r"[^a-z0-9]", "", species.lower())
    suffixes = [
        'alola', 'galar', 'hisui', 'paldea',
        'megax', 'megay', 'mega',
        'primal', 'origin', 'therian', 'incarnate',
        'ice', 'shadow', 'rapid', 'single', 'crowned',
        'heat', 'wash', 'frost', 'fan', 'mow',
        'sky', 'land', 'trash', 'sandy',
        'attack', 'defense', 'speed',
        'bloodmoon', 'wellspring', 'hearthflame', 'cornerstone',
        'stellar', 'terastal',
        'bug', 'dark', 'dragon', 'electric', 'fairy', 'fighting', 'fire',
        'flying', 'ghost', 'grass', 'ground', 'poison',
        'psychic', 'rock', 'steel', 'water',
    ]
    for s in suffixes:
        if name.endswith(s) and len(name) > len(s):
            name = name[:-len(s)] + '-' + s
            break
    return name


def _sprite_url(species: str, animated: bool = False) -> str:
    """Get Showdown sprite URL for a Pokemon."""
    name = _sprite_id(species)
    if animated:
        return f"https://play.pokemonshowdown.com/sprites/ani/{name}.gif"
    return f"https://play.pokemonshowdown.com/sprites/gen5/{name}.png"


def _parse_showdown_paste(paste: str) -> list[dict]:
    """Parse a Pokemon Showdown team paste into structured data."""
    team = []
    current = None

    for line in paste.strip().split("\n"):
        line = line.strip()
        if not line:
            if current:
                team.append(current)
                current = None
            continue

        if current is None:
            current = {"species": "", "ability": "", "item": "", "moves": [],
                       "evs": {}, "ivs": {}, "nature": "", "tera_type": ""}
            # First line: "Pokemon @ Item" or "Nickname (Pokemon) @ Item"
            if " @ " in line:
                parts = line.split(" @ ", 1)
                current["item"] = parts[1].strip()
                name_part = parts[0].strip()
            else:
                name_part = line.strip()

            if "(" in name_part and ")" in name_part:
                # Nickname (Species)
                current["species"] = name_part.split("(")[1].split(")")[0].strip()
            else:
                current["species"] = name_part.strip()
                # Remove gender suffix
                if current["species"].endswith(" (M)") or current["species"].endswith(" (F)"):
                    current["species"] = current["species"][:-4].strip()
        elif line.startswith("Ability:"):
            current["ability"] = line.split(":", 1)[1].strip()
        elif line.startswith("Tera Type:"):
            current["tera_type"] = line.split(":", 1)[1].strip()
        elif line.startswith("EVs:"):
            ev_str = line.split(":", 1)[1].strip()
            for part in ev_str.split("/"):
                part = part.strip()
                if " " in part:
                    val, stat = part.rsplit(" ", 1)
                    stat_map = {"HP": "hp", "Atk": "atk", "Def": "def",
                                "SpA": "spa", "SpD": "spd", "Spe": "spe"}
                    if stat in stat_map:
                        try:
                            current["evs"][stat_map[stat]] = int(val.strip())
                        except ValueError:
                            pass
        elif line.startswith("IVs:"):
            iv_str = line.split(":", 1)[1].strip()
            for part in iv_str.split("/"):
                part = part.strip()
                if " " in part:
                    val, stat = part.rsplit(" ", 1)
                    stat_map = {"HP": "hp", "Atk": "atk", "Def": "def",
                                "SpA": "spa", "SpD": "spd", "Spe": "spe"}
                    if stat in stat_map:
                        try:
                            current["ivs"][stat_map[stat]] = int(val.strip())
                        except ValueError:
                            pass
        elif line.endswith("Nature"):
            current["nature"] = line.replace("Nature", "").strip()
        elif line.startswith("- "):
            current["moves"].append(line[2:].strip())

    if current:
        team.append(current)

    return team


def _team_to_showdown_paste(team: list[PokemonSet]) -> str:
    """Convert a team to Pokemon Showdown paste format."""
    lines = []
    for p in team:
        # Header line
        header = p.species
        if p.item:
            header += f" @ {p.item}"
        lines.append(header)

        if p.ability:
            lines.append(f"Ability: {p.ability}")
        if p.tera_type:
            lines.append(f"Tera Type: {p.tera_type}")
        if p.evs:
            stat_names = {"hp": "HP", "atk": "Atk", "def": "Def",
                          "spa": "SpA", "spd": "SpD", "spe": "Spe"}
            ev_parts = []
            for stat, val in p.evs.items():
                if val > 0 and stat in stat_names:
                    ev_parts.append(f"{val} {stat_names[stat]}")
            if ev_parts:
                lines.append(f"EVs: {' / '.join(ev_parts)}")
        # IVs — only show when non-default (not all 31)
        ivs = getattr(p, 'ivs', None)
        if ivs and isinstance(ivs, dict):
            stat_names_iv = {"hp": "HP", "atk": "Atk", "def": "Def",
                             "spa": "SpA", "spd": "SpD", "spe": "Spe"}
            iv_parts = []
            for stat, val in ivs.items():
                if val is not None and val != 31 and stat in stat_names_iv:
                    iv_parts.append(f"{val} {stat_names_iv[stat]}")
            if iv_parts:
                lines.append(f"IVs: {' / '.join(iv_parts)}")
        if p.nature:
            lines.append(f"{p.nature} Nature")
        for move in p.moves:
            lines.append(f"- {move}")
        lines.append("")

    return "\n".join(lines)
