"""FastAPI server for FutureSightML."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .state import AppState
from ..teambuilder.evaluator import TeamEvaluator
from ..teambuilder.genetic import GeneticTeamBuilder
from ..teambuilder.analysis import TeamAnalyzer
from ..utils.logging_config import setup_logging

log = setup_logging("INFO")

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the GUI
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
    species: str
    ability: str = ""
    item: str = ""
    moves: list[str] = Field(default_factory=list)
    tera_type: str = ""
    evs: dict[str, int] = Field(default_factory=dict)
    nature: str = ""
    level: int = 100


class TeamRequest(BaseModel):
    team: list[PokemonSet]


class GenerateRequest(BaseModel):
    format_id: str = "gen9ou"
    n_results: int = 5
    population: int = 200
    generations: int = 300
    mutation_rate: float = 0.15


class EvaluateRequest(BaseModel):
    format_id: str = "gen9ou"
    team: list[PokemonSet]


class BattleRequest(BaseModel):
    format_id: str = "gen9ou"
    team1: list[PokemonSet]
    team2: list[PokemonSet]


class ImportRequest(BaseModel):
    format_id: str = "gen9ou"
    paste: str


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
    if not fs.pokemon_pool:
        raise HTTPException(400, "No Pokemon pool available.")

    evaluator = TeamEvaluator(predictor=fs.ensemble, meta_teams=fs.meta_teams)
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

    teams = []
    for r in results:
        team_data = []
        for p in r["team"]:
            team_data.append({
                "species": p.get("species", ""),
                "ability": p.get("ability", ""),
                "item": p.get("item", ""),
                "moves": p.get("moves", []),
                "tera_type": p.get("tera_type", ""),
                "evs": p.get("evs", {}),
                "nature": p.get("nature", ""),
            })
        teams.append({
            "pokemon": team_data,
            "predicted_winrate": round(r["fitness"] * 100, 2),
            "generation_found": r["generation"],
        })

    return {"format": fmt, "teams": teams}


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

    # Get moves this Pokemon can learn
    learnable_moves = []
    for move_id, move_data in app_state.pkmn_data.moves.items():
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
    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    team = [p.model_dump() for p in req.team]
    result = analyzer.speed_tier_analysis(team)
    return {"format": req.format_id, "speed_tiers": result}


@app.post("/api/team/analyze/archetype")
async def analyze_archetype(req: EvaluateRequest):
    """Detect the team's playstyle archetype."""
    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    team = [p.model_dump() for p in req.team]
    result = analyzer.detect_archetype(team)
    return {"format": req.format_id, **result}


@app.post("/api/team/analyze/tera")
async def analyze_tera(req: EvaluateRequest):
    """Suggest optimal tera types for each team member."""
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
    analyzer = TeamAnalyzer(pokemon_data=app_state.pkmn_data)
    team = [p.model_dump() for p in req.team]
    result = analyzer.coverage_analysis(team)
    return {"format": req.format_id, **result}


@app.post("/api/team/analyze/full")
async def full_analysis(req: EvaluateRequest):
    """Run all analysis tools on a team at once."""
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

    result = {
        "format": fmt,
        "speed_tiers": speed,
        "archetype": archetype,
        "tera_suggestions": tera,
        "coverage": coverage,
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
# Helpers
# ---------------------------------------------------------------------------

def _sprite_url(species: str, animated: bool = False) -> str:
    """Get Showdown sprite URL for a Pokemon."""
    name = species.lower().replace(" ", "").replace("'", "").replace(".", "")
    # Handle common forme names
    name = name.replace("-mega", "-mega").replace("-alola", "-alola")
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
                       "evs": {}, "nature": "", "tera_type": ""}
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
                        current["evs"][stat_map[stat]] = int(val.strip())
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
        if p.nature:
            lines.append(f"{p.nature} Nature")
        for move in p.moves:
            lines.append(f"- {move}")
        lines.append("")

    return "\n".join(lines)
