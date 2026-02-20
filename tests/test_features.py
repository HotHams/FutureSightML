"""Tests for feature extraction."""

import numpy as np
from showdown.data.features import FeatureExtractor


SAMPLE_BATTLE = {
    "team1": [
        {"species": "garchomp", "ability": "roughskin", "item": "choicescarf",
         "moves": ["earthquake", "outrage", "stoneedge", "fireblast"]},
        {"species": "clefable", "ability": "magicguard", "item": "leftovers",
         "moves": ["moonblast", "softboiled", "stealthrock", "thunderwave"]},
    ],
    "team2": [
        {"species": "ironvaliant", "ability": "quarkdrive", "item": "boosterenergy",
         "moves": ["closecombat", "moonblast", "knockoff", "swordsdance"]},
        {"species": "gholdengo", "ability": "goodasgold", "item": "choicespecs",
         "moves": ["shadowball", "makeitrain", "focusblast", "trick"]},
    ],
    "winner": 1,
}


class TestFeatureExtractor:
    def setup_method(self):
        self.fe = FeatureExtractor()
        self.fe.build_vocab_from_battles([SAMPLE_BATTLE])

    def test_vocab_built(self):
        sizes = self.fe.vocab_sizes
        assert sizes["species"] > 0
        assert sizes["moves"] > 0
        assert sizes["items"] > 0
        assert sizes["abilities"] > 0

    def test_team_to_indices(self):
        indices = self.fe.team_to_indices(SAMPLE_BATTLE["team1"])
        assert indices["species"].shape == (6,)
        assert indices["moves"].shape == (6, 4)
        assert indices["items"].shape == (6,)
        assert indices["abilities"].shape == (6,)

        # First two slots should have non-zero species
        assert indices["species"][0] > 0
        assert indices["species"][1] > 0
        # Remaining slots should be 0 (padding)
        assert indices["species"][2] == 0

    def test_battle_to_tensors(self):
        tensors = self.fe.battle_to_tensors(SAMPLE_BATTLE)
        assert "team1_species" in tensors
        assert "team2_species" in tensors
        assert tensors["label"] == 1.0

    def test_label_flips(self):
        battle_p2_wins = dict(SAMPLE_BATTLE)
        battle_p2_wins["winner"] = 2
        tensors = self.fe.battle_to_tensors(battle_p2_wins)
        assert tensors["label"] == 0.0

    def test_unknown_species(self):
        """Unknown species should map to index 0."""
        team = [{"species": "totally_fake_mon", "moves": [], "item": None, "ability": None}]
        indices = self.fe.team_to_indices(team)
        assert indices["species"][0] == 0
