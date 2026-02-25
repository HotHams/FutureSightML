"""Configuration loader."""

import sys
from pathlib import Path
from typing import Any
import yaml


_CONFIG_CACHE: dict | None = None


def get_data_root() -> Path:
    """Return the root directory for bundled data files.

    In a PyInstaller bundle, data files are extracted to sys._MEIPASS.
    In normal mode, this is the project root (parent of showdown/).
    """
    if getattr(sys, '_MEIPASS', None):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def resolve_data_path(relative_path: str | Path) -> Path:
    """Resolve a relative data path using the appropriate root.

    Works in both normal development and PyInstaller-bundled modes.
    """
    return get_data_root() / relative_path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration file."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE

    if path is None:
        # Try _MEIPASS first (PyInstaller), then walk up, then CWD
        search = resolve_data_path("config.yaml")
        if not search.exists():
            search = Path(__file__).resolve().parent.parent / "config.yaml"
        if not search.exists():
            search = Path.cwd() / "config.yaml"
        path = search

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _CONFIG_CACHE = cfg
    return cfg
