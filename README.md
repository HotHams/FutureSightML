# FutureSightML

**ML-powered Pokemon team builder that predicts win rates from team composition alone.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Formats](https://img.shields.io/badge/formats-61-orange.svg)](#supported-formats)

FutureSightML uses a neural transformer encoder with a pairwise matchup matrix branch, combined with an XGBoost ensemble, to evaluate Pokemon teams before a single move is made. It ships pre-trained models for **61 competitive formats across Generations 1-9** and includes a retro-styled desktop GUI, a FastAPI backend, and a genetic algorithm team builder.

![FutureSightML GUI](docs/screenshot_post.png)

## Key Numbers

Pre-game win prediction from team composition alone (held-out test set, no rating leakage):

Select formats shown (61 total):

| Format | Neural AUC | XGB AUC |
|---|---|---|
| Gen 1 OU | 0.8034 | 0.8023 |
| Gen 3 OU | 0.8590 | 0.8276 |
| Gen 4 OU | 0.8531 | 0.8413 |
| Gen 5 OU | 0.7577 | 0.7429 |
| Gen 6 OU | 0.7640 | 0.7649 |
| Gen 7 OU | 0.7959 | 0.7449 |
| Gen 7 Ubers | 0.8212 | 0.7797 |
| Gen 8 OU | 0.7611 | 0.7223 |
| Gen 8 Ubers | 0.8302 | 0.8047 |
| Gen 8 VGC 2020 | 0.8008 | 0.7807 |
| Gen 9 OU | 0.7598 | 0.6876 |
| Gen 9 Ubers | 0.8119 | 0.7517 |
| Gen 9 AG | 0.8448 | 0.7717 |
| Gen 9 Monotype | 0.8231 | 0.7905 |
| Gen 9 VGC 2026 | 0.7351 | 0.7117 |

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

### Desktop App (Windows)

[Download from HuggingFace](https://huggingface.co/HotHams/FutureSightML/resolve/main/FutureSightML-win-x64.zip) (~3 GB). Extract, run `FutureSightML.exe`. Models download automatically on first launch.

Requires Windows 10+ and an NVIDIA GPU (CUDA) for best performance.

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

- **Team generator** — A genetic algorithm evolves teams to maximize predicted win rate against the metagame
- **Win rate prediction** — Paste or build a team, see its predicted win rate and matchup spread
- **Deep analysis** — Archetype detection, team roles, win conditions, game plan, speed tiers, type coverage, Tera suggestions, and top threats
- **Head-to-head** — Pit any two teams against each other for a win probability breakdown
- **Damage calculator** — Calculate damage between any two Pokemon with full move/item/ability context
- **Metagame overview** — Usage statistics, top Pokemon, common sets, items, and abilities per format
- **Cross-format tier list** — Compare Pokemon viability across all supported formats at a glance
- **Battle simulator** — Monte Carlo battle simulation with turn-by-turn replay
- **Import/Export** — Paste Showdown teams in, export generated teams out
- **Full API** — FastAPI backend with auto-generated docs at `/docs`

## Supported Formats

61 formats with pre-trained models across Generations 1-9, including OU, UU, RU, NU, Ubers, VGC, Doubles OU, LC, Monotype, National Dex, 1v1, AG, and more. See `config.yaml` for the full list.

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

## License

MIT
