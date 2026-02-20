"""Loader for static Pokemon data from the Showdown data repository."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import aiohttp

log = logging.getLogger("showdown.data.pokemon_data")

DATA_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# URLs for raw Showdown data (TypeScript source)
_SHOWDOWN_BASE = "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data"
_ENDPOINTS = {
    "pokedex": f"{_SHOWDOWN_BASE}/pokedex.ts",
    "moves": f"{_SHOWDOWN_BASE}/moves.ts",
    "items": f"{_SHOWDOWN_BASE}/items.ts",
    "abilities": f"{_SHOWDOWN_BASE}/abilities.ts",
    "typechart": f"{_SHOWDOWN_BASE}/typechart.ts",
    "formats": f"{_SHOWDOWN_BASE}/formats-data.ts",
}


def _to_id(name: str) -> str:
    """Convert a Pokemon name to its Showdown ID form."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _parse_ts_object(text: str) -> dict[str, Any]:
    """Best-effort parse of a Showdown TypeScript data export into a Python dict.

    The TS files export a large object literal. We extract it and use a lenient
    JSON-like parse (handling trailing commas, unquoted keys, single-line
    comments, etc.).
    """
    # Remove single-line comments
    text = re.sub(r"//[^\n]*", "", text)
    # Remove multi-line comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Find the main object: everything after the first '=' and '{' up to the
    # matching closing '};'
    match = re.search(r"=\s*(\{)", text)
    if not match:
        raise ValueError("Could not locate object literal in TS source")

    start = match.start(1)
    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    obj_text = text[start:end]

    # Convert TS object literal to valid JSON:
    # 1. Quote unquoted keys
    obj_text = re.sub(r'(?<=[{,\n])\s*([a-zA-Z_]\w*)\s*:', r' "\1":', obj_text)
    # 2. Remove trailing commas before } or ]
    obj_text = re.sub(r",\s*([}\]])", r"\1", obj_text)
    # 3. Replace single quotes with double quotes (for string values)
    obj_text = obj_text.replace("'", '"')
    # 4. Handle template literals (backticks) - just replace with double quotes
    obj_text = obj_text.replace("`", '"')
    # 5. Remove any remaining TypeScript type annotations
    obj_text = re.sub(r"as\s+\w+[\[\]]*", "", obj_text)

    try:
        return json.loads(obj_text)
    except json.JSONDecodeError as e:
        log.warning("JSON parse failed, attempting line-by-line recovery: %s", e)
        return _fallback_parse(obj_text)


def _fallback_parse(text: str) -> dict[str, Any]:
    """Fallback parser: extract top-level keys and their objects individually."""
    result = {}
    # Find top-level key: value pairs where value is an object
    pattern = re.compile(r'"(\w+)":\s*\{', re.MULTILINE)
    for m in pattern.finditer(text):
        key = m.group(1)
        start = m.end() - 1  # the opening brace
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        obj_str = text[start:end]
        try:
            result[key] = json.loads(obj_str)
        except json.JSONDecodeError:
            continue
    return result


class PokemonDataLoader:
    """Downloads and provides access to Pokemon game data from the Showdown repo."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or DATA_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pokedex: dict[str, Any] = {}
        self.moves: dict[str, Any] = {}
        self.items: dict[str, Any] = {}
        self.abilities: dict[str, Any] = {}
        self.formats_data: dict[str, Any] = {}
        self._loaded = False

    async def load(self, force_download: bool = False) -> None:
        """Download and parse all data files."""
        async with aiohttp.ClientSession() as session:
            for name, url in _ENDPOINTS.items():
                cache_file = self.cache_dir / f"{name}.ts"

                if cache_file.exists() and not force_download:
                    raw = cache_file.read_text(encoding="utf-8")
                else:
                    log.info("Downloading %s from Showdown repo...", name)
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            log.error("Failed to download %s: HTTP %d", name, resp.status)
                            continue
                        raw = await resp.text()
                    cache_file.write_text(raw, encoding="utf-8")

                parsed = _parse_ts_object(raw)
                setattr(self, name if name != "formats" else "formats_data", parsed)
                log.info("Parsed %s: %d entries", name, len(parsed))

        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("PokemonDataLoader not loaded. Call await loader.load() first.")

    def get_pokemon(self, name: str) -> dict[str, Any] | None:
        """Look up a Pokemon by name or ID."""
        self.ensure_loaded()
        key = _to_id(name)
        return self.pokedex.get(key)

    def get_base_stats(self, name: str) -> dict[str, int] | None:
        pkmn = self.get_pokemon(name)
        if pkmn is None:
            return None
        return pkmn.get("baseStats")

    def get_types(self, name: str) -> list[str]:
        pkmn = self.get_pokemon(name)
        if pkmn is None:
            return []
        return pkmn.get("types", [])

    def get_move(self, name: str) -> dict[str, Any] | None:
        self.ensure_loaded()
        return self.moves.get(_to_id(name))

    def get_item(self, name: str) -> dict[str, Any] | None:
        self.ensure_loaded()
        return self.items.get(_to_id(name))

    def get_ability(self, name: str) -> dict[str, Any] | None:
        self.ensure_loaded()
        return self.abilities.get(_to_id(name))

    def get_all_species(self) -> list[str]:
        """Return all species IDs in the pokedex."""
        self.ensure_loaded()
        return list(self.pokedex.keys())

    def get_all_moves(self) -> list[str]:
        self.ensure_loaded()
        return list(self.moves.keys())

    def get_all_items(self) -> list[str]:
        self.ensure_loaded()
        return list(self.items.keys())

    def get_all_abilities(self) -> list[str]:
        self.ensure_loaded()
        return list(self.abilities.keys())

    def species_to_idx(self) -> dict[str, int]:
        """Build a species -> index mapping for embeddings."""
        self.ensure_loaded()
        return {name: i + 1 for i, name in enumerate(sorted(self.pokedex.keys()))}

    def move_to_idx(self) -> dict[str, int]:
        self.ensure_loaded()
        return {name: i + 1 for i, name in enumerate(sorted(self.moves.keys()))}

    def item_to_idx(self) -> dict[str, int]:
        self.ensure_loaded()
        return {name: i + 1 for i, name in enumerate(sorted(self.items.keys()))}

    def ability_to_idx(self) -> dict[str, int]:
        self.ensure_loaded()
        return {name: i + 1 for i, name in enumerate(sorted(self.abilities.keys()))}

    def get_tier_pokemon(self, tier: str) -> list[str]:
        """Return species available in a given Smogon tier.

        Uses the formats-data file which contains tier assignments.
        Falls back to returning all species if formats data is missing.
        """
        self.ensure_loaded()
        if not self.formats_data:
            return list(self.pokedex.keys())

        tier_upper = tier.upper()
        tier_map = {
            "OU": ["OU"],
            "UU": ["OU", "UU"],
            "RU": ["OU", "UU", "RU"],
            "NU": ["OU", "UU", "RU", "NU"],
            "PU": ["OU", "UU", "RU", "NU", "PU"],
            "UBERS": ["Uber"],
        }
        allowed_tiers = tier_map.get(tier_upper, [tier_upper])

        result = []
        for species_id, data in self.formats_data.items():
            pkmn_tier = data.get("tier", "")
            if pkmn_tier in allowed_tiers or pkmn_tier.replace("(", "").replace(")", "") in allowed_tiers:
                result.append(species_id)
        return result
