"""Tests for the battle simulator."""

import random
from unittest.mock import MagicMock

import pytest

from showdown.simulator.battle_sim import (
    BattlePokemon,
    BattleSimulator,
    BattleSide,
    HazardState,
    MonteCarloSimulator,
    StatusCondition,
    _infer_evs,
    _infer_nature,
)
from showdown.utils.constants import type_effectiveness_against


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_data():
    """Minimal mock of PokemonDataLoader with real-ish data."""
    data = MagicMock()
    data._loaded = True

    _pokedex = {
        "greatusk": {
            "name": "Great Tusk",
            "types": ["Ground", "Fighting"],
            "baseStats": {"hp": 115, "atk": 131, "def": 131, "spa": 53, "spd": 53, "spe": 87},
        },
        "dragapult": {
            "name": "Dragapult",
            "types": ["Dragon", "Ghost"],
            "baseStats": {"hp": 88, "atk": 120, "def": 75, "spa": 100, "spd": 75, "spe": 142},
        },
        "gholdengo": {
            "name": "Gholdengo",
            "types": ["Steel", "Ghost"],
            "baseStats": {"hp": 87, "atk": 60, "def": 95, "spa": 133, "spd": 91, "spe": 84},
        },
        "toxapex": {
            "name": "Toxapex",
            "types": ["Poison", "Water"],
            "baseStats": {"hp": 50, "atk": 63, "def": 152, "spa": 53, "spd": 142, "spe": 35},
        },
        "ironvaliant": {
            "name": "Iron Valiant",
            "types": ["Fairy", "Fighting"],
            "baseStats": {"hp": 74, "atk": 130, "def": 90, "spa": 120, "spd": 60, "spe": 116},
        },
        "landorustherian": {
            "name": "Landorus-Therian",
            "types": ["Ground", "Flying"],
            "baseStats": {"hp": 89, "atk": 145, "def": 90, "spa": 105, "spd": 80, "spe": 91},
        },
    }

    _moves = {
        "earthquake": {
            "name": "Earthquake", "type": "Ground", "category": "Physical",
            "basePower": 100, "accuracy": 100, "priority": 0,
        },
        "closecombat": {
            "name": "Close Combat", "type": "Fighting", "category": "Physical",
            "basePower": 120, "accuracy": 100, "priority": 0,
        },
        "shadowball": {
            "name": "Shadow Ball", "type": "Ghost", "category": "Special",
            "basePower": 80, "accuracy": 100, "priority": 0,
        },
        "dracometeor": {
            "name": "Draco Meteor", "type": "Dragon", "category": "Special",
            "basePower": 130, "accuracy": 90, "priority": 0,
        },
        "makeitshadowsneak": {
            "name": "Shadow Sneak", "type": "Ghost", "category": "Physical",
            "basePower": 40, "accuracy": 100, "priority": 1,
        },
        "stealthrock": {
            "name": "Stealth Rock", "type": "Rock", "category": "Status",
            "basePower": 0, "accuracy": True, "priority": 0,
            "target": "foeSide",
        },
        "recover": {
            "name": "Recover", "type": "Normal", "category": "Status",
            "basePower": 0, "accuracy": True, "priority": 0,
            "flags": {"heal": 1},
        },
        "swordsdance": {
            "name": "Swords Dance", "type": "Normal", "category": "Status",
            "basePower": 0, "accuracy": True, "priority": 0,
            "boosts": {"atk": 2}, "target": "self",
        },
        "thunderwave": {
            "name": "Thunder Wave", "type": "Electric", "category": "Status",
            "basePower": 0, "accuracy": 90, "priority": 0,
            "status": "par",
        },
        "flamethrower": {
            "name": "Flamethrower", "type": "Fire", "category": "Special",
            "basePower": 90, "accuracy": 100, "priority": 0,
            "secondary": {"chance": 10, "status": "brn"},
        },
        "moonblast": {
            "name": "Moonblast", "type": "Fairy", "category": "Special",
            "basePower": 95, "accuracy": 100, "priority": 0,
        },
        "uturn": {
            "name": "U-turn", "type": "Bug", "category": "Physical",
            "basePower": 70, "accuracy": 100, "priority": 0,
        },
        "spikes": {
            "name": "Spikes", "type": "Ground", "category": "Status",
            "basePower": 0, "accuracy": True, "priority": 0,
            "target": "foeSide",
        },
        "rapidspin": {
            "name": "Rapid Spin", "type": "Normal", "category": "Physical",
            "basePower": 50, "accuracy": 100, "priority": 0,
        },
        "drainingkiss": {
            "name": "Draining Kiss", "type": "Fairy", "category": "Special",
            "basePower": 50, "accuracy": 100, "priority": 0,
            "drain": [3, 4],
        },
    }

    def get_pokemon(name):
        from showdown.data.pokemon_data import _to_id
        return _pokedex.get(_to_id(name))

    def get_move(name):
        from showdown.data.pokemon_data import _to_id
        return _moves.get(_to_id(name))

    data.get_pokemon = get_pokemon
    data.get_move = get_move
    data.pokedex = _pokedex
    data.moves = _moves

    return data


def _make_pset(species, moves, ability="", item="", tera_type=""):
    return {
        "species": species,
        "moves": moves,
        "ability": ability,
        "item": item,
        "tera_type": tera_type,
        "level": 100,
        "evs": {},
        "nature": "",
    }


def _make_team1(mock_data):
    return [
        _make_pset("Great Tusk", ["earthquake", "closecombat", "stealthrock", "rapidspin"],
                    ability="protosynthesis", item="leftovers"),
        _make_pset("Dragapult", ["dracometeor", "shadowball", "flamethrower", "uturn"],
                    ability="infiltrator", item="choicespecs"),
        _make_pset("Gholdengo", ["shadowball", "flamethrower", "thunderwave", "recover"],
                    ability="goodasgold", item="leftovers"),
    ]


def _make_team2(mock_data):
    return [
        _make_pset("Iron Valiant", ["moonblast", "closecombat", "shadowball", "swordsdance"],
                    ability="quarkdrive", item="lifeorb"),
        _make_pset("Landorus-Therian", ["earthquake", "uturn", "stealthrock", "closecombat"],
                    ability="intimidate", item="leftovers"),
        _make_pset("Toxapex", ["recover", "thunderwave", "spikes", "flamethrower"],
                    ability="regenerator", item="blacksludge"),
    ]


# ---------------------------------------------------------------------------
# Tests: BattlePokemon
# ---------------------------------------------------------------------------

class TestBattlePokemon:
    def test_stat_computation(self, mock_data):
        sim = BattleSimulator(mock_data)
        pkmn = sim.build_pokemon(_make_pset(
            "Great Tusk", ["earthquake"], item="leftovers",
        ))
        assert pkmn.species == "greattusk"
        assert pkmn.max_hp > 0
        assert pkmn.current_hp == pkmn.max_hp
        assert pkmn.stats["atk"] > 0
        assert pkmn.alive

    def test_take_damage(self, mock_data):
        sim = BattleSimulator(mock_data)
        pkmn = sim.build_pokemon(_make_pset("Great Tusk", ["earthquake"]))
        initial_hp = pkmn.current_hp
        actual = pkmn.take_damage(50)
        assert actual == 50
        assert pkmn.current_hp == initial_hp - 50
        assert pkmn.alive

    def test_faint(self, mock_data):
        sim = BattleSimulator(mock_data)
        pkmn = sim.build_pokemon(_make_pset("Great Tusk", ["earthquake"]))
        pkmn.take_damage(9999)
        assert pkmn.current_hp == 0
        assert pkmn.is_fainted
        assert not pkmn.alive

    def test_heal(self, mock_data):
        sim = BattleSimulator(mock_data)
        pkmn = sim.build_pokemon(_make_pset("Great Tusk", ["earthquake"]))
        pkmn.take_damage(100)
        healed = pkmn.heal(50)
        assert healed == 50

    def test_heal_caps_at_max(self, mock_data):
        sim = BattleSimulator(mock_data)
        pkmn = sim.build_pokemon(_make_pset("Great Tusk", ["earthquake"]))
        pkmn.take_damage(10)
        healed = pkmn.heal(9999)
        assert healed == 10
        assert pkmn.current_hp == pkmn.max_hp

    def test_stat_stages(self, mock_data):
        sim = BattleSimulator(mock_data)
        pkmn = sim.build_pokemon(_make_pset("Great Tusk", ["earthquake"]))
        base_atk = pkmn.effective_stat("atk")
        pkmn.stat_stages["atk"] = 2
        boosted = pkmn.effective_stat("atk")
        assert boosted > base_atk

    def test_burn_halves_atk(self, mock_data):
        sim = BattleSimulator(mock_data)
        pkmn = sim.build_pokemon(_make_pset("Great Tusk", ["earthquake"]))
        normal_atk = pkmn.effective_stat("atk")
        pkmn.status = StatusCondition.BURN
        burned_atk = pkmn.effective_stat("atk")
        assert burned_atk == normal_atk // 2

    def test_tera_changes_types(self, mock_data):
        sim = BattleSimulator(mock_data)
        pkmn = sim.build_pokemon(_make_pset("Dragapult", ["shadowball"], tera_type="Fire"))
        assert pkmn.effective_types() == ["Dragon", "Ghost"]
        pkmn.is_terastallized = True
        assert pkmn.effective_types() == ["Fire"]


# ---------------------------------------------------------------------------
# Tests: BattleSimulator damage
# ---------------------------------------------------------------------------

class TestDamageCalc:
    def test_stab_bonus(self, mock_data):
        sim = BattleSimulator(mock_data, rng=random.Random(42))
        attacker = sim.build_pokemon(_make_pset("Great Tusk", ["earthquake"]))
        defender = sim.build_pokemon(_make_pset("Gholdengo", ["shadowball"]))
        eq_data = mock_data.get_move("earthquake")
        # Ground vs Steel/Ghost — Ground is super-effective against Steel (2x),
        # neutral against Ghost, so 2x total. STAB Ground from Great Tusk.
        damage = sim._calc_damage(attacker, defender, eq_data)
        assert damage > 0

    def test_immunity(self, mock_data):
        sim = BattleSimulator(mock_data, rng=random.Random(42))
        attacker = sim.build_pokemon(_make_pset("Great Tusk", ["earthquake"]))
        defender = sim.build_pokemon(_make_pset("Dragapult", ["shadowball"]))
        # Fighting vs Ghost = 0x
        cc_data = mock_data.get_move("closecombat")
        damage = sim._calc_damage(attacker, defender, cc_data)
        assert damage == 0

    def test_ability_immunity(self, mock_data):
        sim = BattleSimulator(mock_data, rng=random.Random(42))
        attacker = sim.build_pokemon(_make_pset("Great Tusk", ["earthquake"]))
        defender = sim.build_pokemon(_make_pset("Landorus-Therian", ["earthquake"]))
        # Landorus-T has no Levitate in our mock, but let's set it
        defender.ability = "levitate"
        eq_data = mock_data.get_move("earthquake")
        damage = sim._calc_damage(attacker, defender, eq_data)
        assert damage == 0


# ---------------------------------------------------------------------------
# Tests: Full simulation
# ---------------------------------------------------------------------------

class TestBattleSimulation:
    def test_simulation_completes(self, mock_data):
        sim = BattleSimulator(mock_data, rng=random.Random(42))
        team1 = _make_team1(mock_data)
        team2 = _make_team2(mock_data)
        result = sim.simulate(team1, team2)
        assert result["winner"] in (1, 2)
        assert result["turns"] > 0
        assert 0 <= result["team1_remaining"] <= 3
        assert 0 <= result["team2_remaining"] <= 3

    def test_simulation_deterministic(self, mock_data):
        """Same seed should give same result."""
        team1 = _make_team1(mock_data)
        team2 = _make_team2(mock_data)

        sim1 = BattleSimulator(mock_data, rng=random.Random(99))
        r1 = sim1.simulate(team1, team2)

        sim2 = BattleSimulator(mock_data, rng=random.Random(99))
        r2 = sim2.simulate(team1, team2)

        assert r1["winner"] == r2["winner"]
        assert r1["turns"] == r2["turns"]

    def test_simulation_max_turns(self, mock_data):
        """Simulation should stop at MAX_TURNS."""
        sim = BattleSimulator(mock_data, rng=random.Random(42))
        sim.MAX_TURNS = 10
        team1 = _make_team1(mock_data)
        team2 = _make_team2(mock_data)
        result = sim.simulate(team1, team2)
        assert result["turns"] <= 10


# ---------------------------------------------------------------------------
# Tests: Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_monte_carlo_runs(self, mock_data):
        mc = MonteCarloSimulator(mock_data, n_simulations=10)
        team1 = _make_team1(mock_data)
        team2 = _make_team2(mock_data)
        results = mc.estimate_win_rate(team1, team2)

        assert results["n_simulations"] == 10
        assert results["team1_wins"] + results["team2_wins"] + results["ties"] == 10
        assert 0 <= results["team1_win_rate"] <= 1
        assert 0 <= results["team2_win_rate"] <= 1
        assert results["avg_turns"] > 0

    def test_monte_carlo_custom_n(self, mock_data):
        mc = MonteCarloSimulator(mock_data, n_simulations=5)
        team1 = _make_team1(mock_data)
        team2 = _make_team2(mock_data)
        results = mc.estimate_win_rate(team1, team2, n=20)
        assert results["n_simulations"] == 20


# ---------------------------------------------------------------------------
# Tests: Entry hazards
# ---------------------------------------------------------------------------

class TestHazards:
    def test_stealth_rock_damage(self, mock_data):
        sim = BattleSimulator(mock_data, rng=random.Random(42))
        pkmn = sim.build_pokemon(_make_pset("Great Tusk", ["earthquake"]))
        side = BattleSide(team=[pkmn])
        side.hazards.stealth_rock = True
        opp = BattleSide(team=[sim.build_pokemon(_make_pset("Dragapult", ["shadowball"]))])

        initial_hp = pkmn.current_hp
        sim._apply_switch_in(side, opp)
        # Ground/Fighting takes neutral from Rock (1x), so 1/8 max HP
        expected = max(1, int(pkmn.max_hp * 1.0 / 8))
        assert pkmn.current_hp == initial_hp - expected

    def test_heavy_duty_boots_blocks_hazards(self, mock_data):
        sim = BattleSimulator(mock_data, rng=random.Random(42))
        pkmn = sim.build_pokemon(_make_pset("Great Tusk", ["earthquake"], item="Heavy-Duty Boots"))
        side = BattleSide(team=[pkmn])
        side.hazards.stealth_rock = True
        opp = BattleSide(team=[sim.build_pokemon(_make_pset("Dragapult", ["shadowball"]))])

        initial_hp = pkmn.current_hp
        sim._apply_switch_in(side, opp)
        assert pkmn.current_hp == initial_hp  # No damage!


# ---------------------------------------------------------------------------
# Tests: Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_infer_evs_physical(self, mock_data):
        evs = _infer_evs(
            {"hp": 115, "atk": 131, "def": 131, "spa": 53, "spd": 53, "spe": 87},
            ["earthquake", "closecombat"],
            mock_data,
        )
        assert evs.get("atk", 0) == 252
        assert evs.get("spe", 0) == 252

    def test_infer_evs_special(self, mock_data):
        evs = _infer_evs(
            {"hp": 87, "atk": 60, "def": 95, "spa": 133, "spd": 91, "spe": 84},
            ["shadowball", "flamethrower"],
            mock_data,
        )
        assert evs.get("spa", 0) == 252

    def test_infer_nature_physical(self, mock_data):
        nature = _infer_nature(
            {"hp": 115, "atk": 131, "def": 131, "spa": 53, "spd": 53, "spe": 87},
            {"atk": 252, "spe": 252, "hp": 6},
            ["earthquake", "closecombat"],
            mock_data,
        )
        assert nature == "Jolly"

    def test_infer_nature_special(self, mock_data):
        nature = _infer_nature(
            {"hp": 87, "atk": 60, "def": 95, "spa": 133, "spd": 91, "spe": 84},
            {"spa": 252, "spe": 252, "hp": 6},
            ["shadowball", "flamethrower"],
            mock_data,
        )
        assert nature == "Timid"
