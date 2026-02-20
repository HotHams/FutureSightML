"""Loader for static Pokemon data from the Showdown data repository.

Uses clean JSON endpoints where available, falls back to TS parsing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import aiohttp

log = logging.getLogger("showdown.data.pokemon_data")

DATA_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

# Clean JSON endpoints (served by the Showdown client)
_JSON_ENDPOINTS = {
    "pokedex": "https://play.pokemonshowdown.com/data/pokedex.json",
    "moves": "https://play.pokemonshowdown.com/data/moves.json",
}

# TypeScript sources (for data without JSON endpoints)
_SHOWDOWN_BASE = "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data"
_TS_ENDPOINTS = {
    "items": f"{_SHOWDOWN_BASE}/items.ts",
    "abilities": f"{_SHOWDOWN_BASE}/abilities.ts",
    "typechart": f"{_SHOWDOWN_BASE}/typechart.ts",
    "formats": f"{_SHOWDOWN_BASE}/formats-data.ts",
}


def _to_id(name: str) -> str:
    """Convert a Pokemon name to its Showdown ID form."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _parse_ts_object(text: str) -> dict[str, Any]:
    """Parse a Showdown TypeScript data export into a Python dict.

    Handles unquoted keys (including numeric), trailing commas,
    single quotes, template literals, and TS type annotations.
    """
    # Remove single-line comments (but not inside strings)
    text = re.sub(r"//[^\n]*", "", text)
    # Remove multi-line comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Find the main object
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
    # 1. Replace single quotes with double quotes
    obj_text = obj_text.replace("'", '"')
    # 2. Handle template literals
    obj_text = obj_text.replace("`", '"')
    # 3. Remove TS type annotations (as Type, as const, etc.)
    obj_text = re.sub(r"\bas\s+\w+[\[\]]*", "", obj_text)
    # 4. Quote ALL unquoted keys (alphabetic, numeric, underscore)
    #    Match keys preceded by { , or newline that aren't already quoted
    obj_text = re.sub(
        r'(?<=[{,\n\t])\s*([a-zA-Z_]\w*)\s*:',
        r' "\1":',
        obj_text,
    )
    # Also handle numeric keys like  0: "Overgrow"
    obj_text = re.sub(
        r'(?<=[{,\n\t])\s*(\d+)\s*:',
        r' "\1":',
        obj_text,
    )
    # 5. Remove trailing commas before } or ]
    obj_text = re.sub(r",\s*([}\]])", r"\1", obj_text)
    # 6. Remove any remaining problematic constructs
    # Handle spread operators or other JS features
    obj_text = re.sub(r"\.\.\.\w+", '""', obj_text)

    try:
        return json.loads(obj_text)
    except json.JSONDecodeError as e:
        log.warning("Full JSON parse failed (%s), using entry-by-entry recovery", e)
        return _fallback_parse(obj_text)


def _fallback_parse(text: str) -> dict[str, Any]:
    """Fallback parser: extract top-level key:object entries one at a time."""
    result = {}
    # Match top-level entries: "key": { ... }
    # We need to handle both quoted and potentially unquoted keys
    pattern = re.compile(r'["\s](\w+)"\s*:\s*\{', re.MULTILINE)
    positions = [(m.group(1), m.end() - 1) for m in pattern.finditer(text)]

    for key, brace_start in positions:
        depth = 0
        end = brace_start
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        obj_str = text[brace_start:end]
        try:
            result[key] = json.loads(obj_str)
        except json.JSONDecodeError:
            # Try fixing common issues in this single entry
            fixed = re.sub(r",\s*([}\]])", r"\1", obj_str)
            try:
                result[key] = json.loads(fixed)
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
            # Phase 1: Load from clean JSON endpoints
            for name, url in _JSON_ENDPOINTS.items():
                cache_file = self.cache_dir / f"{name}.json"

                if cache_file.exists() and not force_download:
                    raw = cache_file.read_text(encoding="utf-8")
                else:
                    log.info("Downloading %s from Showdown client (JSON)...", name)
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            log.error("Failed to download %s: HTTP %d", name, resp.status)
                            continue
                        raw = await resp.text()
                    cache_file.write_text(raw, encoding="utf-8")

                parsed = json.loads(raw)
                setattr(self, name, parsed)
                log.info("Loaded %s: %d entries (JSON)", name, len(parsed))

            # Phase 2: Load from TS sources (items, abilities, typechart, formats)
            for name, url in _TS_ENDPOINTS.items():
                cache_file = self.cache_dir / f"{name}.ts"

                if cache_file.exists() and not force_download:
                    raw = cache_file.read_text(encoding="utf-8")
                else:
                    log.info("Downloading %s from Showdown repo (TS)...", name)
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            log.error("Failed to download %s: HTTP %d", name, resp.status)
                            continue
                        raw = await resp.text()
                    cache_file.write_text(raw, encoding="utf-8")

                parsed = _parse_ts_object(raw)
                attr_name = name if name != "formats" else "formats_data"
                setattr(self, attr_name, parsed)
                log.info("Parsed %s: %d entries (TS)", name, len(parsed))

        # Supplement abilities from pokedex data (TS parser misses most due to JS functions)
        if len(self.abilities) < 100 and self.pokedex:
            extracted = self._extract_abilities_from_pokedex()
            if len(extracted) > len(self.abilities):
                log.info(
                    "Supplementing abilities from pokedex: %d -> %d entries",
                    len(self.abilities), len(extracted),
                )
                self.abilities = extracted

        self._loaded = True
        log.info(
            "Data load complete: %d species, %d moves, %d items, %d abilities",
            len(self.pokedex), len(self.moves), len(self.items), len(self.abilities),
        )

    def ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("PokemonDataLoader not loaded. Call await loader.load() first.")

    def get_pokemon(self, name: str) -> dict[str, Any] | None:
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

    def _extract_abilities_from_pokedex(self) -> dict[str, Any]:
        """Extract ability entries from pokedex data as a fallback."""
        abilities = {}
        for species_data in self.pokedex.values():
            for ability_name in species_data.get("abilities", {}).values():
                aid = _to_id(ability_name)
                if aid and aid not in abilities:
                    abilities[aid] = {"name": ability_name}
        return abilities

    def get_tier_pokemon(self, tier: str) -> list[str]:
        """Return species available in a given Smogon tier."""
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
