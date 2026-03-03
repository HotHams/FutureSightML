"""Tests for format constraints."""

from showdown.teambuilder.constraints import FormatConstraints


class TestFormatConstraints:
    def test_legal_team(self):
        constraints = FormatConstraints("gen9ou")
        team = [
            {"species": "garchomp", "ability": "roughskin", "item": "choicescarf",
             "moves": ["earthquake", "dragonclaw", "swordsdance", "stealthrock"]},
            {"species": "clefable", "ability": "magicguard", "item": "leftovers",
             "moves": ["moonblast", "softboiled", "calmmind", "thunderwave"]},
            {"species": "ferrothorn", "ability": "ironbarbs", "item": "rockyhelmet",
             "moves": ["stealthrock", "leechseed", "powerwhip", "knockoff"]},
            {"species": "rotomwash", "ability": "levitate", "item": "heavydutyboots",
             "moves": ["voltswitch", "hydropump", "willowisp", "painsplit"]},
            {"species": "dragapult", "ability": "clearbody", "item": "choicespecs",
             "moves": ["shadowball", "dracometeor", "fireblast", "uturn"]},
            {"species": "greattusk", "ability": "protosynthesis", "item": "boosterenergy",
             "moves": ["earthquake", "closecombat", "icespinner", "rapidspin"]},
        ]
        legal, violations = constraints.is_team_legal(team)
        assert legal, f"Violations: {violations}"

    def test_species_clause(self):
        constraints = FormatConstraints("gen9ou")
        team = [
            {"species": "garchomp", "moves": []},
            {"species": "garchomp", "moves": []},  # duplicate
        ]
        legal, violations = constraints.is_team_legal(team)
        assert not legal
        assert any("Species Clause" in v for v in violations)

    def test_banned_pokemon(self):
        constraints = FormatConstraints("gen9ou")
        team = [
            {"species": "koraidon", "moves": []},  # banned in OU
        ]
        legal, violations = constraints.is_team_legal(team)
        assert not legal
        assert any("Banned" in v for v in violations)

    def test_vgc_item_clause(self):
        constraints = FormatConstraints("gen9vgc2024regg")
        team = [
            {"species": "garchomp", "item": "leftovers", "moves": []},
            {"species": "clefable", "item": "leftovers", "moves": []},  # dupe item
        ]
        legal, violations = constraints.is_team_legal(team)
        assert not legal
        assert any("Item Clause" in v for v in violations)

    def test_smogon_allows_dupe_items(self):
        constraints = FormatConstraints("gen9ou")
        team = [
            {"species": "garchomp", "item": "leftovers",
             "moves": ["earthquake", "dragonclaw", "swordsdance", "stealthrock"]},
            {"species": "clefable", "item": "leftovers",
             "moves": ["moonblast", "softboiled", "calmmind", "thunderwave"]},
        ]
        legal, violations = constraints.is_team_legal(team)
        assert legal

    def test_base_species_normalization(self):
        assert FormatConstraints._base_species("rotomwash") == "rotom"
        assert FormatConstraints._base_species("charizardmegax") == "charizard"
        assert FormatConstraints._base_species("garchomp") == "garchomp"

    def test_empty_team(self):
        constraints = FormatConstraints("gen9ou")
        legal, violations = constraints.is_team_legal([])
        assert not legal
