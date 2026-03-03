"""Tests for damage calculator and gen-aware move categories."""

from showdown.utils.constants import get_move_category, unify_special_stat
from showdown.data.damage_calc import estimate_damage_pct


class TestGetMoveCategory:
    """Test gen-aware move category resolution."""

    def test_gen1_thunderbolt_is_special(self):
        move = {"category": "Special", "type": "Electric", "basePower": 90}
        assert get_move_category(move, gen=1) == "Special"

    def test_gen1_earthquake_is_physical(self):
        move = {"category": "Physical", "type": "Ground", "basePower": 100}
        assert get_move_category(move, gen=1) == "Physical"

    def test_gen1_psychic_is_special(self):
        # In Gen 1-3, Psychic type = Special (determined by type)
        move = {"category": "Special", "type": "Psychic", "basePower": 90}
        assert get_move_category(move, gen=1) == "Special"

    def test_gen1_toxic_is_status(self):
        move = {"category": "Status", "type": "Poison", "basePower": 0}
        assert get_move_category(move, gen=1) == "Status"

    def test_gen3_shadow_ball_is_physical(self):
        # Ghost type is Physical in Gen 1-3 (before the physical/special split)
        # Even though Gen 4+ data says it's Special
        move = {"category": "Special", "type": "Ghost", "basePower": 80}
        assert get_move_category(move, gen=3) == "Physical"

    def test_gen9_shadow_ball_is_special(self):
        # In Gen 4+, Shadow Ball uses its per-move category (Special)
        move = {"category": "Special", "type": "Ghost", "basePower": 80}
        assert get_move_category(move, gen=9) == "Special"

    def test_gen3_fire_blast_is_special(self):
        move = {"category": "Special", "type": "Fire", "basePower": 110}
        assert get_move_category(move, gen=3) == "Special"

    def test_gen3_cross_chop_is_physical(self):
        # Fighting type = Physical in Gen 1-3
        move = {"category": "Physical", "type": "Fighting", "basePower": 100}
        assert get_move_category(move, gen=3) == "Physical"

    def test_gen4_uses_per_move_category(self):
        # Gen 4+: category is per-move, not per-type
        # A hypothetical Physical Fire move stays Physical
        move = {"category": "Physical", "type": "Fire", "basePower": 75}
        assert get_move_category(move, gen=4) == "Physical"

    def test_gen2_ice_beam_is_special(self):
        move = {"category": "Special", "type": "Ice", "basePower": 90}
        assert get_move_category(move, gen=2) == "Special"

    def test_gen1_normal_is_physical(self):
        move = {"category": "Physical", "type": "Normal", "basePower": 120}
        assert get_move_category(move, gen=1) == "Physical"

    def test_status_unchanged_any_gen(self):
        move = {"category": "Status", "type": "Fire", "basePower": 0}
        for gen in range(1, 10):
            assert get_move_category(move, gen=gen) == "Status"


class TestUnifySpecialStat:
    """Test Gen 1 unified Special stat helper."""

    def test_gen1_unifies_spa_spd(self):
        stats = {"hp": 55, "atk": 50, "def": 65, "spa": 135, "spd": 95, "spe": 120}
        unified = unify_special_stat(stats, gen=1)
        assert unified["spa"] == 135
        assert unified["spd"] == 135  # SpD unified to SpA value

    def test_gen2_keeps_separate(self):
        stats = {"hp": 55, "atk": 50, "def": 65, "spa": 135, "spd": 95, "spe": 120}
        result = unify_special_stat(stats, gen=2)
        assert result["spa"] == 135
        assert result["spd"] == 95  # Stays separate

    def test_gen9_keeps_separate(self):
        stats = {"hp": 55, "atk": 50, "def": 65, "spa": 135, "spd": 95, "spe": 120}
        result = unify_special_stat(stats, gen=9)
        assert result["spa"] == 135
        assert result["spd"] == 95

    def test_gen1_preserves_other_stats(self):
        stats = {"hp": 100, "atk": 80, "def": 70, "spa": 120, "spd": 90, "spe": 110}
        unified = unify_special_stat(stats, gen=1)
        assert unified["hp"] == 100
        assert unified["atk"] == 80
        assert unified["def"] == 70
        assert unified["spe"] == 110


class TestEstimateDamagePctGenAware:
    """Test that damage calculations use correct stats based on gen."""

    def _make_pokemon(self, atk=100, spa=100, dfn=100, spd=100, hp=300,
                      types=None, stab_types=None):
        return {
            "atk": atk, "spa": spa, "def": dfn, "spd": spd, "hp_actual": hp,
            "types": types or [], "stab_types": stab_types or set(),
            "item_id": "", "ability_id": "",
            "item_effects": {}, "ability_effects": {},
            "atk_item_mult": 1.0, "spa_item_mult": 1.0,
            "def_item_mult": 1.0, "spd_item_mult": 1.0,
            "damage_mult": 1.0, "stab_mult": 1.5,
        }

    def test_gen3_shadow_ball_uses_atk(self):
        # Ghost type is Physical in Gen 3 -> should use Atk stat
        attacker = self._make_pokemon(atk=200, spa=50)
        defender = self._make_pokemon(dfn=100, spd=100)
        move = {"basePower": 80, "category": "Special", "type": "Ghost",
                "name": "Shadow Ball", "flags": {}}

        dmg_gen3 = estimate_damage_pct(attacker, defender, move, gen=3)
        dmg_gen9 = estimate_damage_pct(attacker, defender, move, gen=9)

        # Gen 3: uses Atk (200) -> much higher damage
        # Gen 9: uses SpA (50) -> much lower damage
        assert dmg_gen3 > dmg_gen9

    def test_gen1_fire_blast_uses_spa(self):
        # Fire type is Special in Gen 1 -> should use SpA stat
        attacker = self._make_pokemon(atk=50, spa=200)
        defender = self._make_pokemon(dfn=100, spd=100)
        move = {"basePower": 110, "category": "Special", "type": "Fire",
                "name": "Fire Blast", "flags": {}}

        dmg = estimate_damage_pct(attacker, defender, move, gen=1)
        # Should use SpA (200) against SpD (100) since Fire is Special in Gen 1
        assert dmg > 0

    def test_gen9_fire_blast_uses_spa(self):
        # Fire Blast is Special in Gen 9 too (same per-move category)
        attacker = self._make_pokemon(atk=50, spa=200)
        defender = self._make_pokemon(dfn=100, spd=100)
        move = {"basePower": 110, "category": "Special", "type": "Fire",
                "name": "Fire Blast", "flags": {}}

        dmg = estimate_damage_pct(attacker, defender, move, gen=9)
        assert dmg > 0

    def test_gen1_earthquake_uses_atk(self):
        # Ground type is Physical in Gen 1 -> should use Atk stat
        attacker = self._make_pokemon(atk=200, spa=50)
        defender = self._make_pokemon(dfn=80, spd=200)
        move = {"basePower": 100, "category": "Physical", "type": "Ground",
                "name": "Earthquake", "flags": {}}

        dmg = estimate_damage_pct(attacker, defender, move, gen=1)
        # Uses Atk (200) against Def (80) -> significant damage
        assert dmg > 0

    def test_status_move_returns_zero(self):
        attacker = self._make_pokemon()
        defender = self._make_pokemon()
        move = {"basePower": 0, "category": "Status", "type": "Poison",
                "name": "Toxic", "flags": {}}

        for gen in [1, 3, 9]:
            assert estimate_damage_pct(attacker, defender, move, gen=gen) == 0.0
