"""Tests for type chart and constants."""

from showdown.utils.constants import (
    type_effectiveness,
    type_effectiveness_against,
    calc_stat,
    TYPES,
    NATURES,
)


class TestTypeChart:
    def test_neutral(self):
        assert type_effectiveness("Normal", "Normal") == 1.0

    def test_super_effective(self):
        assert type_effectiveness("Fire", "Grass") == 2.0
        assert type_effectiveness("Water", "Fire") == 2.0
        assert type_effectiveness("Electric", "Water") == 2.0

    def test_not_very_effective(self):
        assert type_effectiveness("Fire", "Water") == 0.5
        assert type_effectiveness("Grass", "Fire") == 0.5

    def test_immune(self):
        assert type_effectiveness("Normal", "Ghost") == 0.0
        assert type_effectiveness("Electric", "Ground") == 0.0
        assert type_effectiveness("Ground", "Flying") == 0.0
        assert type_effectiveness("Ghost", "Normal") == 0.0
        assert type_effectiveness("Psychic", "Dark") == 0.0
        assert type_effectiveness("Dragon", "Fairy") == 0.0
        assert type_effectiveness("Fighting", "Ghost") == 0.0
        assert type_effectiveness("Poison", "Steel") == 0.0

    def test_dual_type(self):
        # Fire vs Grass/Steel = 2.0 * 2.0 = 4.0
        eff = type_effectiveness_against("Fire", ["Grass", "Steel"])
        assert eff == 4.0

        # Electric vs Water/Flying = 2.0 * 2.0 = 4.0
        eff = type_effectiveness_against("Electric", ["Water", "Flying"])
        assert eff == 4.0

        # Ground vs Flying = 0.0 (immune regardless of second type)
        eff = type_effectiveness_against("Ground", ["Flying", "Steel"])
        assert eff == 0.0

    def test_all_types_exist(self):
        assert len(TYPES) == 18
        assert "Fairy" in TYPES

    def test_fairy_interactions(self):
        assert type_effectiveness("Fairy", "Dragon") == 2.0
        assert type_effectiveness("Fairy", "Fighting") == 2.0
        assert type_effectiveness("Fairy", "Dark") == 2.0
        assert type_effectiveness("Fairy", "Steel") == 0.5
        assert type_effectiveness("Fairy", "Poison") == 0.5
        assert type_effectiveness("Fairy", "Fire") == 0.5


class TestStatCalc:
    def test_hp_stat(self):
        # Garchomp: base HP 108, 31 IV, 252 EV, level 100
        # floor((2*108 + 31 + 63) * 100/100) + 100 + 10 = 310 + 110 = 420
        hp = calc_stat(108, 31, 252, 100, 1.0, is_hp=True)
        assert hp == 420

    def test_attack_stat(self):
        # Garchomp: base Atk 130, 31 IV, 252 EV, level 100, neutral nature
        # floor((2*130 + 31 + 63) * 100/100 + 5) * 1.0 = 359
        atk = calc_stat(130, 31, 252, 100, 1.0, is_hp=False)
        assert atk == 359

    def test_shedinja_hp(self):
        hp = calc_stat(1, 31, 0, 100, 1.0, is_hp=True)
        assert hp == 1

    def test_nature_boost(self):
        # Adamant nature boosts Atk by 1.1
        atk_neutral = calc_stat(100, 31, 252, 100, 1.0, is_hp=False)
        atk_boosted = calc_stat(100, 31, 252, 100, 1.1, is_hp=False)
        assert atk_boosted > atk_neutral

    def test_natures_dict(self):
        assert "Adamant" in NATURES
        assert NATURES["Adamant"]["atk"] == 1.1
        assert NATURES["Adamant"]["spa"] == 0.9
        assert NATURES["Hardy"] == {}
