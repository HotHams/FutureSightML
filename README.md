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
| Gen 9 OU | 0.7592 | 0.6872 | 90/10 |
| Gen 9 UU | 0.7570 | 0.6927 | 70/30 |
| Gen 9 RU | 0.7743 | 0.7219 | 80/20 |
| Gen 9 NU | 0.7515 | 0.7047 | 85/15 |
| Gen 9 Ubers | 0.8079 | 0.7519 | 90/10 |
| Gen 9 VGC 2026 | 0.7391 | 0.7106 | 80/20 |
| Gen 9 Doubles OU | 0.7555 | 0.7189 | 75/25 |
| Gen 1 OU | 0.8134 | 0.8023 | 65/35 |
| Gen 2 OU | 0.7367 | 0.7545 | 45/55 |
| Gen 3 OU | 0.8544 | 0.8273 | 65/35 |

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

```bash
# Clone and install
git clone https://github.com/HotHams/FutureSightML.git
cd FutureSightML
pip install -e .

# Start the server
python scripts/run_server.py

# Open http://localhost:8000 in your browser
```

## Desktop App

Download pre-built binaries from [Releases](https://github.com/HotHams/FutureSightML/releases), or build from source:

```bash
python scripts/build_exe.py
```

This builds the PyInstaller backend + Electron frontend into a standalone desktop app.

## Training Your Own Models

```bash
# 1. Scrape replays from Pokemon Showdown
python scripts/scrape.py --format gen9ou --count 10000

# 2. (Optional) Import bulk data from HuggingFace
python scripts/import_huggingface.py --gen 9

# 3. Train models for all configured formats
python scripts/train_all_formats.py

# 4. Start the server with your models
python scripts/run_server.py
```

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
│   ├── train.py               # Train a single format
│   ├── train_all_formats.py   # Train all configured formats
│   ├── scrape.py              # Scrape replays
│   ├── run_server.py          # Start the API server
│   ├── build_exe.py           # Build desktop app
│   └── import_huggingface.py  # Import bulk replay dataset
├── data/                      # SQLite DB, model checkpoints, vocab files
├── tests/                     # Test suite
├── config.yaml                # Format and training configuration
└── FutureSightML.spec         # PyInstaller build spec
```

## API

The FastAPI server provides auto-generated docs at [http://localhost:8000/docs](http://localhost:8000/docs).

Key endpoints:
- `POST /api/team/generate` - Generate an optimized team for a format
- `POST /api/team/analyze/full` - Analyze a team's predicted win rate and matchups
- `GET /api/meta/{format}` - Get metagame usage statistics
- `POST /api/battle/simulate` - Monte Carlo battle simulation

## Supported Formats

129 formats across Generations 1-9, including OU, UU, RU, NU, Ubers, VGC, Doubles OU, LC, Monotype, and more. See `config.yaml` for the full list.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run existing tests (`pytest tests/ -v`)
4. Submit a pull request

## License

MIT
