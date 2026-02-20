"""Tests for the replay parser."""

import pytest
from showdown.scraper.replay_parser import ReplayParser


SAMPLE_LOG = """|j|☆Alice
|j|☆Bob
|player|p1|Alice|1|1200
|player|p2|Bob|2|1300
|teamsize|p1|6
|teamsize|p2|6
|gametype|singles
|gen|9
|tier|[Gen 9] OU
|rule|Species Clause
|poke|p1|Garchomp, M|
|poke|p1|Clefable, F|
|poke|p1|Ferrothorn, M|
|poke|p1|Rotom-Wash|
|poke|p1|Dragapult, M|
|poke|p1|Great Tusk|
|poke|p2|Iron Valiant|
|poke|p2|Gholdengo|
|poke|p2|Kingambit, M|
|poke|p2|Corviknight, F|
|poke|p2|Garganacl, M|
|poke|p2|Dragonite, M|
|start
|switch|p1a: Garchomp|Garchomp, M|100/100
|switch|p2a: Iron Valiant|Iron Valiant|100/100
|turn|1
|move|p1a: Garchomp|Earthquake|p2a: Iron Valiant
|-damage|p2a: Iron Valiant|45/100
|move|p2a: Iron Valiant|Moonblast|p1a: Garchomp
|-damage|p1a: Garchomp|68/100
|turn|2
|move|p1a: Garchomp|Scale Shot|p2a: Iron Valiant
|-damage|p2a: Iron Valiant|0 fnt
|faint|p2a: Iron Valiant
|switch|p2a: Gholdengo|Gholdengo|100/100
|-ability|p2a: Gholdengo|Good as Gold
|turn|3
|switch|p1a: Ferrothorn|Ferrothorn, M|100/100
|-item|p1a: Ferrothorn|Leftovers
|move|p2a: Gholdengo|Shadow Ball|p1a: Ferrothorn
|-damage|p1a: Ferrothorn|82/100
|turn|4
|move|p1a: Ferrothorn|Stealth Rock|
|move|p2a: Gholdengo|Focus Blast|p1a: Ferrothorn
|-damage|p1a: Ferrothorn|0 fnt
|faint|p1a: Ferrothorn
|win|Alice"""


class TestReplayParser:
    def setup_method(self):
        self.parser = ReplayParser()

    def test_parse_players(self):
        result = self.parser.parse(SAMPLE_LOG, "gen9ou")
        assert result.player1 == "Alice"
        assert result.player2 == "Bob"

    def test_parse_winner(self):
        result = self.parser.parse(SAMPLE_LOG, "gen9ou")
        assert result.winner == 1  # Alice wins

    def test_parse_team_preview(self):
        result = self.parser.parse(SAMPLE_LOG, "gen9ou")
        team1_species = {p.species for p in result.team1}
        assert "garchomp" in team1_species
        assert "clefable" in team1_species
        assert "ferrothorn" in team1_species

    def test_parse_team2(self):
        result = self.parser.parse(SAMPLE_LOG, "gen9ou")
        team2_species = {p.species for p in result.team2}
        assert "ironvaliant" in team2_species
        assert "gholdengo" in team2_species
        assert "kingambit" in team2_species

    def test_parse_moves(self):
        result = self.parser.parse(SAMPLE_LOG, "gen9ou")
        garchomp = next(p for p in result.team1 if p.species == "garchomp")
        assert "earthquake" in garchomp.moves
        assert "scaleshot" in garchomp.moves

    def test_parse_ability(self):
        result = self.parser.parse(SAMPLE_LOG, "gen9ou")
        gholdengo = next(p for p in result.team2 if p.species == "gholdengo")
        assert gholdengo.ability == "goodasgold"

    def test_parse_item(self):
        result = self.parser.parse(SAMPLE_LOG, "gen9ou")
        ferrothorn = next(p for p in result.team1 if p.species == "ferrothorn")
        assert ferrothorn.item == "leftovers"

    def test_is_valid(self):
        result = self.parser.parse(SAMPLE_LOG, "gen9ou")
        assert result.is_valid()

    def test_empty_log_invalid(self):
        result = self.parser.parse("", "gen9ou")
        assert not result.is_valid()

    def test_to_dict(self):
        result = self.parser.parse(SAMPLE_LOG, "gen9ou")
        for pkmn in result.team1:
            d = pkmn.to_dict()
            assert "species" in d
            assert "moves" in d
            assert isinstance(d["moves"], list)
