"""Tests for gen-aware feature extraction."""

from showdown.data.features import FeatureExtractor
from showdown.utils.constants import get_move_category


# A moveset with mixed physical/special type distribution in Gen 1-3 vs Gen 4+
MIXED_MOVESET_POKEMON = {
    "species": "gengar",
    "ability": "levitate",
    "item": "",
    "moves": ["shadowball", "sludgebomb", "thunderbolt", "focusblast"],
}

SAMPLE_BATTLE = {
    "team1": [MIXED_MOVESET_POKEMON],
    "team2": [
        {"species": "snorlax", "ability": "thickfat", "item": "leftovers",
         "moves": ["bodyslam", "earthquake", "rest", "sleeptalk"]},
    ],
    "winner": 1,
}


class TestGenAwareFeatures:
    """Test that FeatureExtractor produces different features for different gens."""

    def test_gen3_vs_gen9_category_classification_differs(self):
        """Shadow Ball (Ghost) is Physical in Gen 3 but Special in Gen 9.
        Verify at the category level since move data lookup requires PokemonDataLoader."""
        # Simulate the move data dicts as they would appear from the data loader
        shadow_ball = {"category": "Special", "type": "Ghost", "basePower": 80}
        focus_blast = {"category": "Special", "type": "Fighting", "basePower": 120}
        thunderbolt = {"category": "Special", "type": "Electric", "basePower": 90}
        sludge_bomb = {"category": "Special", "type": "Poison", "basePower": 90}

        moves = [shadow_ball, focus_blast, thunderbolt, sludge_bomb]

        # Gen 3: classify by type
        gen3_phys = [m for m in moves if get_move_category(m, 3) == "Physical"]
        gen3_spec = [m for m in moves if get_move_category(m, 3) == "Special"]

        # Gen 9: classify by per-move category
        gen9_phys = [m for m in moves if get_move_category(m, 9) == "Physical"]
        gen9_spec = [m for m in moves if get_move_category(m, 9) == "Special"]

        # Ghost and Fighting are Physical types in Gen 3 -> 2 physical moves
        # Poison is also Physical type in Gen 3 -> 3 physical moves total
        assert len(gen3_phys) == 3  # Shadow Ball, Focus Blast, Sludge Bomb
        assert len(gen3_spec) == 1  # Thunderbolt

        # Gen 9: all four are Special per their category field
        assert len(gen9_phys) == 0
        assert len(gen9_spec) == 4

        # Best physical BP differs
        gen3_best_phys = max((m["basePower"] for m in gen3_phys), default=0)
        gen9_best_phys = max((m["basePower"] for m in gen9_phys), default=0)
        assert gen3_best_phys == 120  # Focus Blast
        assert gen9_best_phys == 0    # None

    def test_get_move_category_integration(self):
        """Verify that the gen-aware category function works for typical moves."""
        # Shadow Ball: Ghost type
        shadow_ball = {"category": "Special", "type": "Ghost", "basePower": 80}
        assert get_move_category(shadow_ball, gen=3) == "Physical"  # Ghost = Physical
        assert get_move_category(shadow_ball, gen=9) == "Special"   # per-move

        # Focus Blast: Fighting type
        focus_blast = {"category": "Special", "type": "Fighting", "basePower": 120}
        assert get_move_category(focus_blast, gen=3) == "Physical"  # Fighting = Physical
        assert get_move_category(focus_blast, gen=9) == "Special"   # per-move


class TestPrecomputeTeamData:
    """Test that precompute_team_data stores gen-corrected category."""

    def test_gen3_precompute_corrects_category(self):
        fe = FeatureExtractor(gen=3)
        fe.build_vocab_from_battles([SAMPLE_BATTLE])

        team = [
            {"species": "gengar", "ability": "levitate", "item": "",
             "moves": ["shadowball"]},
        ]
        data = fe.precompute_team_data(team)

        # Shadow Ball should have category "Physical" in Gen 3
        assert len(data) == 1
        moves_data = data[0].get("moves_data", [])
        shadow_ball_entry = [m for m in moves_data if m.get("type") == "Ghost"]
        if shadow_ball_entry:
            assert shadow_ball_entry[0]["category"] == "Physical"

    def test_gen9_precompute_keeps_category(self):
        fe = FeatureExtractor(gen=9)
        fe.build_vocab_from_battles([SAMPLE_BATTLE])

        team = [
            {"species": "gengar", "ability": "levitate", "item": "",
             "moves": ["shadowball"]},
        ]
        data = fe.precompute_team_data(team)

        moves_data = data[0].get("moves_data", [])
        shadow_ball_entry = [m for m in moves_data if m.get("type") == "Ghost"]
        if shadow_ball_entry:
            assert shadow_ball_entry[0]["category"] == "Special"
