# FutureSightML

**ML-powered Pokemon team builder that predicts win rates from team composition alone.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Formats](https://img.shields.io/badge/formats-129-orange.svg)](#supported-formats)

FutureSightML uses a neural transformer encoder with a pairwise matchup matrix branch, combined with an XGBoost ensemble, to evaluate Pokemon teams before a single move is made. It supports **129 formats across Generations 1-9** and includes a retro-styled desktop GUI, a FastAPI backend, and a genetic algorithm team builder.

## Key Numbers

Pre-game win prediction from team composition alone (held-out test set, no rating leakage):

| Format | Neural AUC | XGB AUC | Ensemble (N/X) |
|---|---|---|---|
| Gen 9 OU | 0.7588 | 0.6878 | 75/25 |
| Gen 9 UU | 0.7561 | 0.6915 | 70/30 |
| Gen 9 RU | 0.7723 | 0.7221 | 85/15 |
| Gen 9 NU | 0.7471 | 0.7047 | 80/20 |
| Gen 9 Ubers | 0.8140 | 0.7519 | 80/20 |
| Gen 9 VGC 2026 | 0.7371 | 0.7106 | 70/30 |
| Gen 9 Doubles OU | 0.7506 | 0.7196 | 85/15 |
| Gen 1 OU | 0.8098 | 0.8023 | 55/45 |
| Gen 2 OU | 0.7329 | 0.7545 | 25/75 |
| Gen 3 OU | 0.8543 | 0.8272 | 65/35 |

All numbers are team-only AUC on a held-out test set with equalized ratings (no Elo leakage). For context: Dota 2 draft prediction AUC is 0.66-0.71, Hearthstone deck prediction AUC is 0.65-0.68.

## Architecture

```
Team A (6 Pokemon) ──► PokemonEncoder ──► TeamEncoder (3-layer Transformer)
                                                    ├──► CrossAttention
Team B (6 Pokemon) ──► PokemonEncoder ──► TeamEncoder ──┤
                                                    └──► MatchupMatrixBranch (6x6 pairwise)
                                                              │
                                                              ▼
                                                         MatchupHead ──► P(Team A wins)

XGBoost (618+ engineered features) ──────────────────────► Calibrated Ensemble
```

- **Neural model**: Species/move/item/ability embeddings, continuous stat features, 3-layer transformer encoder with cross-attention and a blade-chest matchup matrix decomposition
- **XGBoost**: 618+ hand-engineered features including type coverage, stat distributions, threat matrices, item/ability effects, and dual-rating features
- **Ensemble**: Per-format calibrated weights, saved to JSON
- **Gen-aware**: Type charts, stat formulas, move categories, and learnsets are all generation-correct

## Quick Start

### Desktop App

Download pre-built binaries from [Releases](https://github.com/HotHams/FutureSightML/releases). Runs out of the box — no Python required.

### From Source

```bash
# Clone and install
git clone https://github.com/HotHams/FutureSightML.git
cd FutureSightML
pip install -e .

# Download pre-trained models (~310 MB)
python scripts/download_models.py

# Start the server (opens browser automatically)
python scripts/run_server.py
```

Everything runs locally. Nothing leaves your machine.

## What You Get

- **Win rate prediction** — Paste or build a team, see its predicted win rate against the current metagame
- **Team generator** — A genetic algorithm evolves teams to maximize predicted win rate
- **Team analysis** — Type coverage, speed tiers, threat matchups, archetype classification
- **Battle simulator** — Monte Carlo battle simulation with turn-by-turn replay
- **Full API** — FastAPI backend with auto-generated docs at `/docs`

## Supported Formats

129 formats across Generations 1-9, including OU, UU, RU, NU, Ubers, VGC, Doubles OU, LC, Monotype, National Dex, and more. See `config.yaml` for the full list.

Pre-trained models ship for the 11 most popular formats. All other formats have metagame pools and can be trained with additional replay data.

## Building the Desktop App

```bash
python scripts/build_exe.py
```

This builds the PyInstaller backend + Electron frontend into a standalone desktop app.

## API

The FastAPI server provides auto-generated docs at [http://localhost:8000/docs](http://localhost:8000/docs).

Key endpoints:
- `POST /api/team/generate` - Generate an optimized team for a format
- `POST /api/team/analyze/full` - Analyze a team's predicted win rate and matchups
- `GET /api/meta/{format}` - Get metagame usage statistics
- `POST /api/battle/simulate` - Monte Carlo battle simulation

## Project Structure

```
FutureSightML/
├── showdown/                  # Main Python package
│   ├── models/                # Neural network architectures
│   ├── data/                  # Feature extraction, damage calc, data loading
│   ├── scraper/               # Replay scraping from Showdown API
│   ├── teambuilder/           # Genetic algorithm, evaluation, analysis
│   ├── simulator/             # Battle simulator with Monte Carlo estimation
│   ├── api/                   # FastAPI server and app state
│   └── utils/                 # Constants, logging, type system
├── gui/                       # Electron + React frontend
│   └── static/index.html      # Single-file React app with retro UI
├── scripts/                   # CLI entry points
│   ├── run_server.py          # Start the API server
│   ├── download_models.py     # Download pre-trained models
│   ├── build_exe.py           # Build desktop app
│   └── train_model.py         # Train models (for development)
├── data/                      # Model checkpoints, pool data
├── tests/                     # Test suite (90 tests)
├── config.yaml                # Format and training configuration
└── FutureSightML.spec         # PyInstaller build spec
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run existing tests (`pytest tests/ -v`)
4. Submit a pull request

## Acknowledgments

- **HolidayOugi** for the [pokemon-showdown-replays](https://huggingface.co/datasets/HolidayOugi/pokemon-showdown-replays) dataset (30.5M replays) on HuggingFace
- **Smogon** and the [Pokemon Showdown](https://pokemonshowdown.com/) community for usage statistics and replay data
- **pkmn.cc** for the Showdown data files (pokedex, moves, items, abilities, learnsets)

## License

MIT
