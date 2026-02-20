"""Parser for Pokemon Showdown battle replay logs.

Extracts structured team and battle data from the raw log text format.
"""

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger("showdown.scraper.replay_parser")


def _to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass
class PokemonSet:
    species: str
    ability: str | None = None
    item: str | None = None
    moves: list[str] = field(default_factory=list)
    tera_type: str | None = None
    level: int = 100

    def to_dict(self) -> dict:
        return {
            "species": self.species,
            "ability": self.ability,
            "item": self.item,
            "moves": self.moves[:4],
            "tera_type": self.tera_type,
            "level": self.level,
        }


@dataclass
class BattleResult:
    format: str = ""
    player1: str = ""
    player2: str = ""
    winner: int = 0  # 1, 2, or 0 for tie
    turns: int = 0
    team1: list[PokemonSet] = field(default_factory=list)
    team2: list[PokemonSet] = field(default_factory=list)

    def is_valid(self) -> bool:
        return (
            self.winner in (1, 2)
            and len(self.team1) > 0
            and len(self.team2) > 0
            and self.player1 != ""
            and self.player2 != ""
        )


class ReplayParser:
    """Parse a Pokemon Showdown replay log into structured data."""

    def parse(self, log_text: str, format_id: str = "") -> BattleResult:
        result = BattleResult(format=format_id)

        # Track pokemon per player: player_key -> {nickname_or_species: PokemonSet}
        player_pokemon: dict[str, dict[str, PokemonSet]] = {"p1": {}, "p2": {}}
        # Map nickname to species
        nickname_map: dict[str, str] = {}  # "p1: Nickname" -> species_id
        # Track join order for fallback player name detection
        joined_players: list[str] = []

        for line in log_text.split("\n"):
            line = line.strip()
            if not line or not line.startswith("|"):
                continue

            parts = line.split("|")
            # parts[0] is empty (before first |), parts[1] is the command
            if len(parts) < 2:
                continue
            cmd = parts[1]

            try:
                if cmd == "j" and len(parts) >= 3:
                    # |j|☆username — join event, track as fallback for player names
                    name = parts[2].strip().lstrip("\u2606").strip()  # strip ☆
                    if name and name not in joined_players:
                        joined_players.append(name)

                elif cmd == "player" and len(parts) >= 4:
                    self._handle_player(parts, result)

                elif cmd == "poke" and len(parts) >= 4:
                    self._handle_poke(parts, player_pokemon)

                elif cmd in ("switch", "drag") and len(parts) >= 4:
                    self._handle_switch(parts, player_pokemon, nickname_map)

                elif cmd == "move" and len(parts) >= 4:
                    self._handle_move(parts, player_pokemon, nickname_map)

                elif cmd == "-ability" and len(parts) >= 4:
                    self._handle_ability(parts, player_pokemon, nickname_map)

                elif cmd == "-item" and len(parts) >= 4:
                    self._handle_item(parts, player_pokemon, nickname_map)

                elif cmd == "-enditem" and len(parts) >= 4:
                    self._handle_enditem(parts, player_pokemon, nickname_map)

                elif cmd == "terastallize" and len(parts) >= 4:
                    self._handle_tera(parts, player_pokemon, nickname_map)

                elif cmd == "win" and len(parts) >= 3:
                    winner_name = parts[2].strip()
                    if winner_name and winner_name == result.player1:
                        result.winner = 1
                    elif winner_name and winner_name == result.player2:
                        result.winner = 2
                    else:
                        # Fallback: store winner name, resolve after loop
                        result._winner_name = winner_name

                elif cmd == "tie":
                    result.winner = 0

                elif cmd == "turn" and len(parts) >= 3:
                    try:
                        result.turns = int(parts[2].strip())
                    except ValueError:
                        pass

            except (IndexError, ValueError) as e:
                log.debug("Parse error on line %r: %s", line, e)
                continue

        # Fallback: fill player names from |j| lines if not set
        if not result.player1 and len(joined_players) >= 1:
            result.player1 = joined_players[0]
        if not result.player2 and len(joined_players) >= 2:
            result.player2 = joined_players[1]

        # Resolve winner if not already resolved
        if result.winner == 0 and hasattr(result, "_winner_name"):
            winner_name = result._winner_name
            if winner_name and winner_name == result.player1:
                result.winner = 1
            elif winner_name and winner_name == result.player2:
                result.winner = 2

        # Assemble final teams
        result.team1 = list(player_pokemon["p1"].values())
        result.team2 = list(player_pokemon["p2"].values())

        return result

    def _handle_player(self, parts: list[str], result: BattleResult) -> None:
        player_id = parts[2].strip()  # p1 or p2
        player_name = parts[3].strip()
        if player_id == "p1":
            result.player1 = player_name
        elif player_id == "p2":
            result.player2 = player_name

    def _handle_poke(self, parts: list[str], player_pokemon: dict) -> None:
        """Handle team preview: |poke|p1|Species, L50, M|item"""
        player = parts[2].strip()  # p1 or p2
        if player not in player_pokemon:
            return

        details = parts[3].strip()
        species = self._parse_species_from_details(details)
        level = self._parse_level_from_details(details)

        species_id = _to_id(species)
        if species_id not in player_pokemon[player]:
            player_pokemon[player][species_id] = PokemonSet(
                species=species_id, level=level
            )

    def _handle_switch(self, parts: list[str], player_pokemon: dict,
                       nickname_map: dict) -> None:
        """Handle switch-in: |switch|p1a: Nickname|Species, L100, M|HP"""
        ident = parts[2].strip()  # "p1a: Nickname"
        details = parts[3].strip()  # "Species, L100, M"

        player = ident[:2]  # p1 or p2
        if player not in player_pokemon:
            return

        species = self._parse_species_from_details(details)
        level = self._parse_level_from_details(details)
        species_id = _to_id(species)

        # Map this ident to the species
        nickname_map[ident] = species_id
        # Also map short form without position slot
        short_ident = player + ": " + ident.split(": ", 1)[-1] if ": " in ident else ident
        nickname_map[short_ident] = species_id

        if species_id not in player_pokemon[player]:
            player_pokemon[player][species_id] = PokemonSet(
                species=species_id, level=level
            )

    def _handle_move(self, parts: list[str], player_pokemon: dict,
                     nickname_map: dict) -> None:
        """Handle move usage: |move|p1a: Nickname|Move Name|..."""
        ident = parts[2].strip()
        move_name = parts[3].strip()

        species_id = self._resolve_species(ident, nickname_map)
        if species_id is None:
            return

        player = ident[:2]
        if player not in player_pokemon or species_id not in player_pokemon[player]:
            return

        move_id = _to_id(move_name)
        pkmn = player_pokemon[player][species_id]
        if move_id not in pkmn.moves and len(pkmn.moves) < 4:
            pkmn.moves.append(move_id)

    def _handle_ability(self, parts: list[str], player_pokemon: dict,
                        nickname_map: dict) -> None:
        """|−ability|p1a: Nickname|Ability Name"""
        ident = parts[2].strip()
        ability_name = parts[3].strip()

        species_id = self._resolve_species(ident, nickname_map)
        if species_id is None:
            return

        player = ident[:2]
        if player in player_pokemon and species_id in player_pokemon[player]:
            player_pokemon[player][species_id].ability = _to_id(ability_name)

    def _handle_item(self, parts: list[str], player_pokemon: dict,
                     nickname_map: dict) -> None:
        """|−item|p1a: Nickname|Item Name"""
        ident = parts[2].strip()
        item_name = parts[3].strip()

        species_id = self._resolve_species(ident, nickname_map)
        if species_id is None:
            return

        player = ident[:2]
        if player in player_pokemon and species_id in player_pokemon[player]:
            pkmn = player_pokemon[player][species_id]
            if pkmn.item is None:
                pkmn.item = _to_id(item_name)

    def _handle_enditem(self, parts: list[str], player_pokemon: dict,
                        nickname_map: dict) -> None:
        """|−enditem|p1a: Nickname|Item Name|[from]..."""
        ident = parts[2].strip()
        item_name = parts[3].strip()

        species_id = self._resolve_species(ident, nickname_map)
        if species_id is None:
            return

        player = ident[:2]
        if player in player_pokemon and species_id in player_pokemon[player]:
            pkmn = player_pokemon[player][species_id]
            if pkmn.item is None:
                pkmn.item = _to_id(item_name)

    def _handle_tera(self, parts: list[str], player_pokemon: dict,
                     nickname_map: dict) -> None:
        """|terastallize|p1a: Nickname|Type"""
        ident = parts[2].strip()
        tera_type = parts[3].strip()

        species_id = self._resolve_species(ident, nickname_map)
        if species_id is None:
            return

        player = ident[:2]
        if player in player_pokemon and species_id in player_pokemon[player]:
            player_pokemon[player][species_id].tera_type = tera_type

    def _resolve_species(self, ident: str, nickname_map: dict) -> str | None:
        """Resolve a battle identifier to a species ID."""
        if ident in nickname_map:
            return nickname_map[ident]
        # Try without the position slot (p1a -> p1)
        player = ident[:2]
        short = player + ": " + ident.split(": ", 1)[-1] if ": " in ident else ident
        if short in nickname_map:
            return nickname_map[short]
        return None

    @staticmethod
    def _parse_species_from_details(details: str) -> str:
        """Extract species name from details string like 'Garchomp, L100, M'."""
        # Species is everything before the first comma
        species = details.split(",")[0].strip()
        # Remove forme suffixes that are cosmetic only
        # But keep important formes like -Mega, -Gmax, etc.
        return species

    @staticmethod
    def _parse_level_from_details(details: str) -> int:
        """Extract level from details string. Default 100."""
        match = re.search(r"L(\d+)", details)
        return int(match.group(1)) if match else 100
