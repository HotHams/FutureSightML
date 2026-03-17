"""Game mechanics registry for Pokemon Showdown.

Provides structured lookup tables for item effects, ability effects, and
move categorization, all derived from the actual Showdown data files.

These values are cross-referenced with:
- Pokemon Showdown source code (items.ts, abilities.ts, moves.json)
- Bulbapedia mechanics articles
- Serebii.net item/ability pages

The registries serve as the authoritative source for numeric multipliers
used in damage calculation and feature engineering.
"""

import logging
import re
from typing import Any

log = logging.getLogger("showdown.data.mechanics")


def _to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ===================================================================
# ITEM EFFECTS REGISTRY
# ===================================================================
# Each entry maps item_id -> dict of effects.
# Multipliers come from Showdown's chainModify() values:
#   5324/4096 ≈ 1.3 (Life Orb), 4915/4096 ≈ 1.2 (Expert Belt),
#   4505/4096 ≈ 1.1 (Muscle Band)
# ===================================================================

ITEM_EFFECTS: dict[str, dict[str, Any]] = {
    # --- Choice items (lock holder into one move) ---
    "choiceband":  {"atk_mult": 1.5, "locks_move": True, "category": "choice"},
    "choicespecs": {"spa_mult": 1.5, "locks_move": True, "category": "choice"},
    "choicescarf":  {"spe_mult": 1.5, "locks_move": True, "category": "choice"},

    # --- Offensive damage items ---
    "lifeorb":      {"damage_mult": 1.3, "recoil_frac": 0.1, "category": "offensive"},
    "expertbelt":   {"se_damage_mult": 1.2, "category": "offensive"},
    "metronome":    {"consecutive_mult": True, "category": "offensive"},
    "muscleband":   {"phys_damage_mult": 1.1, "category": "offensive"},
    "wiseglasses":  {"spec_damage_mult": 1.1, "category": "offensive"},
    "scopelens":    {"crit_boost": 1, "category": "offensive"},
    "razorclaw":    {"crit_boost": 1, "category": "offensive"},

    # --- Ability-enhancing offensive items ---
    "punchingglove": {"punch_damage_mult": 1.1, "no_contact": True, "category": "offensive"},
    "loadeddice":    {"multi_hit_bias": True, "category": "offensive"},
    "throatspray":   {"on_sound_move": {"spa_boost": 1}, "single_use": True, "category": "offensive"},

    # --- Type-boosting items (1.2x for matching type) ---
    "charcoal":     {"type_boost": "Fire", "type_mult": 1.2, "category": "type_boost"},
    "mysticwater":  {"type_boost": "Water", "type_mult": 1.2, "category": "type_boost"},
    "miracleseed":  {"type_boost": "Grass", "type_mult": 1.2, "category": "type_boost"},
    "magnet":       {"type_boost": "Electric", "type_mult": 1.2, "category": "type_boost"},
    "nevermeltice": {"type_boost": "Ice", "type_mult": 1.2, "category": "type_boost"},
    "blackbelt":    {"type_boost": "Fighting", "type_mult": 1.2, "category": "type_boost"},
    "poisonbarb":   {"type_boost": "Poison", "type_mult": 1.2, "category": "type_boost"},
    "softsand":     {"type_boost": "Ground", "type_mult": 1.2, "category": "type_boost"},
    "sharpbeak":    {"type_boost": "Flying", "type_mult": 1.2, "category": "type_boost"},
    "twistedspoon": {"type_boost": "Psychic", "type_mult": 1.2, "category": "type_boost"},
    "silverpowder": {"type_boost": "Bug", "type_mult": 1.2, "category": "type_boost"},
    "hardstone":    {"type_boost": "Rock", "type_mult": 1.2, "category": "type_boost"},
    "spelltag":     {"type_boost": "Ghost", "type_mult": 1.2, "category": "type_boost"},
    "dragonfang":   {"type_boost": "Dragon", "type_mult": 1.2, "category": "type_boost"},
    "blackglasses": {"type_boost": "Dark", "type_mult": 1.2, "category": "type_boost"},
    "metalcoat":    {"type_boost": "Steel", "type_mult": 1.2, "category": "type_boost"},
    "silkscarf":    {"type_boost": "Normal", "type_mult": 1.2, "category": "type_boost"},
    "fairyfeather": {"type_boost": "Fairy", "type_mult": 1.2, "category": "type_boost"},

    # --- Defensive stat items ---
    "assaultvest":  {"spd_mult": 1.5, "blocks_status": True, "category": "defensive"},
    "eviolite":     {"def_mult": 1.5, "spd_mult": 1.5, "requires_nfe": True, "category": "defensive"},
    "rockyhelmet":  {"contact_damage": 1/6, "category": "defensive"},

    # --- Survivability items ---
    "focussash":    {"survives_ohko": True, "single_use": True, "full_hp_only": True, "category": "defensive"},
    "focusband":    {"survive_chance": 0.1, "category": "defensive"},

    # --- Recovery items ---
    "leftovers":    {"end_turn_heal": 1/16, "category": "recovery"},
    "blacksludge":  {"end_turn_heal": 1/16, "poison_only": True, "category": "recovery"},
    "shellbell":    {"drain_frac": 1/8, "category": "recovery"},

    # --- Utility / protection items ---
    "heavydutyboots": {"hazard_immune": True, "category": "utility"},
    "airballoon":     {"ground_immune": True, "pops_on_hit": True, "category": "utility"},
    "safetygoggles":  {"powder_immune": True, "weather_immune": True, "category": "utility"},
    "covertcloak":    {"secondary_immune": True, "category": "utility"},
    "shedshell":      {"trap_immune": True, "category": "utility"},

    # --- Activation / conditional items ---
    "weaknesspolicy": {"on_se_hit": {"atk_boost": 2, "spa_boost": 2}, "single_use": True, "category": "offensive"},
    "boosterenergy":  {"boosts_highest_stat": True, "paradox_only": True, "category": "offensive"},

    # --- Terrain seeds ---
    "electricseed":  {"on_terrain": "electric", "def_boost": 1, "single_use": True, "category": "terrain_seed"},
    "grassyseed":    {"on_terrain": "grassy", "def_boost": 1, "single_use": True, "category": "terrain_seed"},
    "mistyseed":     {"on_terrain": "misty", "spd_boost": 1, "single_use": True, "category": "terrain_seed"},
    "psychicseed":   {"on_terrain": "psychic", "spd_boost": 1, "single_use": True, "category": "terrain_seed"},
    "terrainextender": {"extends_terrain": True, "category": "terrain_seed"},

    # --- Common competitive berries ---
    "sitrusberry":  {"heal_threshold": 0.5, "heal_amount": 0.25, "is_berry": True, "category": "berry"},
    "lumberry":     {"cures_status": True, "is_berry": True, "category": "berry"},
    "salacberry":   {"low_hp_boost": {"spe": 1}, "is_berry": True, "category": "berry"},
    "liechiberry":  {"low_hp_boost": {"atk": 1}, "is_berry": True, "category": "berry"},
    "petayaberry":  {"low_hp_boost": {"spa": 1}, "is_berry": True, "category": "berry"},
    "custapberry":  {"priority_boost": True, "is_berry": True, "category": "berry"},
    "aguavberry":   {"heal_threshold": 0.25, "heal_amount": 1/3, "is_berry": True, "category": "berry"},
    "figyberry":    {"heal_threshold": 0.25, "heal_amount": 1/3, "is_berry": True, "category": "berry"},
    "wikiberry":    {"heal_threshold": 0.25, "heal_amount": 1/3, "is_berry": True, "category": "berry"},
    "magoberry":    {"heal_threshold": 0.25, "heal_amount": 1/3, "is_berry": True, "category": "berry"},
    "iapapaberry":  {"heal_threshold": 0.25, "heal_amount": 1/3, "is_berry": True, "category": "berry"},
    "cheriberry":   {"cures": "par", "is_berry": True, "category": "berry"},
    "chestoberry":  {"cures": "slp", "is_berry": True, "category": "berry"},
    "rawstberry":   {"cures": "brn", "is_berry": True, "category": "berry"},
    "aspearberry":  {"cures": "frz", "is_berry": True, "category": "berry"},
    "pechaberry":   {"cures": "psn", "is_berry": True, "category": "berry"},

    # --- Type-resist berries (0.5x on SE hit, single use) ---
    "babiriberry":  {"resists_type": "Steel", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "chartiberry":  {"resists_type": "Rock", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "chopleberry":  {"resists_type": "Fighting", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "cobaberry":    {"resists_type": "Flying", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "colburberry":  {"resists_type": "Dark", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "habanberry":   {"resists_type": "Dragon", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "kasibberry":   {"resists_type": "Ghost", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "kebiaberry":   {"resists_type": "Poison", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "passhoberry":  {"resists_type": "Water", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "payapaberry":  {"resists_type": "Psychic", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "rindoberry":   {"resists_type": "Grass", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "shucaberry":   {"resists_type": "Ground", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "tangaberry":   {"resists_type": "Bug", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "wacanberry":   {"resists_type": "Electric", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "yacheberry":   {"resists_type": "Ice", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "occaberry":    {"resists_type": "Fire", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "roseliberry":  {"resists_type": "Fairy", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},
    "chilanberry":  {"resists_type": "Normal", "resist_mult": 0.5, "is_berry": True, "category": "resist_berry"},

    # --- Mega Stones (Gen 6-7, all 48 canonical) ---
    "venusaurite":       {"category": "mega_stone"},
    "charizarditex":     {"category": "mega_stone"},
    "charizarditey":     {"category": "mega_stone"},
    "blastoisinite":     {"category": "mega_stone"},
    "alakazite":         {"category": "mega_stone"},
    "gengarite":         {"category": "mega_stone"},
    "kangaskhanite":     {"category": "mega_stone"},
    "pinsirite":         {"category": "mega_stone"},
    "gyaradosite":       {"category": "mega_stone"},
    "aerodactylite":     {"category": "mega_stone"},
    "mewtwonitex":       {"category": "mega_stone"},
    "mewtwonitey":       {"category": "mega_stone"},
    "ampharosite":       {"category": "mega_stone"},
    "scizorite":         {"category": "mega_stone"},
    "heracronite":       {"category": "mega_stone"},
    "houndoominite":     {"category": "mega_stone"},
    "tyranitarite":      {"category": "mega_stone"},
    "blazikenite":       {"category": "mega_stone"},
    "gardevoirite":      {"category": "mega_stone"},
    "mawilite":          {"category": "mega_stone"},
    "aggronite":         {"category": "mega_stone"},
    "medichamite":       {"category": "mega_stone"},
    "manectite":         {"category": "mega_stone"},
    "banettite":         {"category": "mega_stone"},
    "absolite":          {"category": "mega_stone"},
    "garchompite":       {"category": "mega_stone"},
    "lucarionite":       {"category": "mega_stone"},
    "abomasite":         {"category": "mega_stone"},
    "latiasite":         {"category": "mega_stone"},
    "latiosite":         {"category": "mega_stone"},
    "swampertite":       {"category": "mega_stone"},
    "sceptilite":        {"category": "mega_stone"},
    "sablenite":         {"category": "mega_stone"},
    "altarianite":       {"category": "mega_stone"},
    "galladite":         {"category": "mega_stone"},
    "audinite":          {"category": "mega_stone"},
    "metagrossite":      {"category": "mega_stone"},
    "sharpedonite":      {"category": "mega_stone"},
    "slowbronite":       {"category": "mega_stone"},
    "steelixite":        {"category": "mega_stone"},
    "pidgeotite":        {"category": "mega_stone"},
    "glalitite":         {"category": "mega_stone"},
    "diancite":          {"category": "mega_stone"},
    "cameruptite":       {"category": "mega_stone"},
    "lopunnite":         {"category": "mega_stone"},
    "salamencite":       {"category": "mega_stone"},
    "beedrillite":       {"category": "mega_stone"},
    # Primal Reversion orbs (functionally equivalent to Mega Stones)
    "redorb":            {"category": "mega_stone"},
    "blueorb":           {"category": "mega_stone"},

    # --- Z-Crystals (Gen 7) — type-specific ---
    "normaliumz":        {"category": "z_crystal"},
    "firiumz":           {"category": "z_crystal"},
    "wateriumz":         {"category": "z_crystal"},
    "electriumz":        {"category": "z_crystal"},
    "grassiumz":         {"category": "z_crystal"},
    "iciumz":            {"category": "z_crystal"},
    "fightiniumz":       {"category": "z_crystal"},
    "poisoniumz":        {"category": "z_crystal"},
    "groundiumz":        {"category": "z_crystal"},
    "flyiniumz":         {"category": "z_crystal"},
    "psychiumz":         {"category": "z_crystal"},
    "buginiumz":         {"category": "z_crystal"},
    "rockiumz":          {"category": "z_crystal"},
    "ghostiumz":         {"category": "z_crystal"},
    "dragoniumz":        {"category": "z_crystal"},
    "darkiniumz":        {"category": "z_crystal"},
    "steeliumz":         {"category": "z_crystal"},
    "fairiumz":          {"category": "z_crystal"},
    # Z-Crystals — species-specific
    "pikaniumz":         {"category": "z_crystal"},
    "aloraichiumz":      {"category": "z_crystal"},
    "decidiumz":         {"category": "z_crystal"},
    "inciniumz":         {"category": "z_crystal"},
    "primariumz":        {"category": "z_crystal"},
    "tapuniumz":         {"category": "z_crystal"},
    "marshadiumz":       {"category": "z_crystal"},
    "snorliumz":         {"category": "z_crystal"},
    "eeviumz":           {"category": "z_crystal"},
    "mewniumz":          {"category": "z_crystal"},
    "pikashuniumz":      {"category": "z_crystal"},
    "lycaniumz":         {"category": "z_crystal"},
    "mimikiumz":         {"category": "z_crystal"},
    "kommoniumz":        {"category": "z_crystal"},
    "solganiumz":        {"category": "z_crystal"},
    "lunaliumz":         {"category": "z_crystal"},
    "ultranecroziumz":   {"category": "z_crystal"},
}


# ===================================================================
# ABILITY EFFECTS REGISTRY
# ===================================================================
# Multiplier values from Showdown's chainModify():
#   5325/4096 ≈ 1.3 (Tough Claws, Sheer Force)
#   5120/4096 ≈ 1.25 (Neuroforce)
#   5448/4096 ≈ 1.33 (Auras)
# ===================================================================

ABILITY_EFFECTS: dict[str, dict[str, Any]] = {
    # --- Offensive power boost abilities ---
    "hugepower":     {"atk_mult": 2.0, "category": "power_boost"},
    "purepower":     {"atk_mult": 2.0, "category": "power_boost"},
    "adaptability":  {"stab_mult": 2.0, "category": "power_boost"},
    "technician":    {"low_bp_mult": 1.5, "bp_threshold": 60, "category": "power_boost"},
    "toughclaws":    {"contact_mult": 1.3, "category": "power_boost"},
    "sheerforce":    {"no_secondary_mult": 1.3, "removes_secondary": True, "category": "power_boost"},
    "strongjaw":     {"bite_mult": 1.5, "category": "power_boost"},
    "ironfist":      {"punch_mult": 1.2, "category": "power_boost"},
    "megalauncher":  {"pulse_mult": 1.5, "category": "power_boost"},
    "skilllink":     {"multi_hit_max": True, "category": "power_boost"},
    "stakeout":      {"switch_in_mult": 2.0, "category": "power_boost"},
    "neuroforce":    {"se_mult": 1.25, "category": "power_boost"},
    "punkrock":      {"sound_mult": 1.3, "sound_resist": 0.5, "category": "power_boost"},
    "analytic":      {"last_move_mult": 1.3, "category": "power_boost"},
    "solarpower":    {"sun_spa_mult": 1.5, "sun_hp_drain": 1/8, "category": "power_boost"},
    "supremeoverlord": {"ko_boost": True, "max_mult": 1.5, "category": "power_boost"},
    "toxicboost":    {"poison_atk_mult": 1.5, "category": "power_boost"},
    "flareboost":    {"burn_spa_mult": 1.5, "category": "power_boost"},
    "guts":          {"status_atk_mult": 1.5, "ignores_burn": True, "category": "power_boost"},
    "swordofruin":   {"opp_def_mult": 0.75, "category": "power_boost"},
    "tabletsofruins": {"opp_atk_mult": 0.75, "category": "power_boost"},
    "vesselofruin":  {"opp_spa_mult": 0.75, "category": "power_boost"},
    "beadsofruin":   {"opp_spd_mult": 0.75, "category": "power_boost"},
    "dragonsmaw":    {"dragon_mult": 1.5, "category": "power_boost"},
    "transistor":    {"electric_mult": 1.3, "category": "power_boost"},
    "steelworker":   {"steel_mult": 1.5, "category": "power_boost"},
    "waterbubble":   {"water_mult": 2.0, "fire_resist": 0.5, "burn_immune": True, "category": "power_boost"},
    "gorilla tactics": {"atk_mult": 1.5, "locks_move": True, "category": "power_boost"},

    # --- Type-changing abilities (Normal → type, 1.2x boost) ---
    "refrigerate":   {"type_change_from": "Normal", "type_change_to": "Ice", "type_change_mult": 1.2, "category": "type_change"},
    "pixilate":      {"type_change_from": "Normal", "type_change_to": "Fairy", "type_change_mult": 1.2, "category": "type_change"},
    "aerilate":      {"type_change_from": "Normal", "type_change_to": "Flying", "type_change_mult": 1.2, "category": "type_change"},
    "galvanize":     {"type_change_from": "Normal", "type_change_to": "Electric", "type_change_mult": 1.2, "category": "type_change"},

    # --- Type immunity abilities ---
    "levitate":      {"type_immunity": "Ground", "category": "immunity"},
    "flashfire":     {"type_immunity": "Fire", "on_absorb_boost": "Fire", "boost_mult": 1.5, "category": "immunity"},
    "voltabsorb":    {"type_immunity": "Electric", "on_absorb_heal": 0.25, "category": "immunity"},
    "waterabsorb":   {"type_immunity": "Water", "on_absorb_heal": 0.25, "category": "immunity"},
    "lightningrod":  {"type_immunity": "Electric", "on_absorb_boost_stat": "spa", "category": "immunity"},
    "stormdrain":    {"type_immunity": "Water", "on_absorb_boost_stat": "spa", "category": "immunity"},
    "motordrive":    {"type_immunity": "Electric", "on_absorb_boost_stat": "spe", "category": "immunity"},
    "sapsipper":     {"type_immunity": "Grass", "on_absorb_boost_stat": "atk", "category": "immunity"},
    "dryskin":       {"type_immunity": "Water", "on_absorb_heal": 0.25, "fire_weakness_mult": 1.25, "category": "immunity"},
    "eartheater":    {"type_immunity": "Ground", "on_absorb_heal": 0.25, "category": "immunity"},
    "wellbakedbody": {"type_immunity": "Fire", "on_absorb_boost_stat": "def", "on_absorb_boost_stages": 2, "category": "immunity"},
    "windrider":     {"wind_immune": True, "on_tailwind_boost": "atk", "category": "immunity"},
    "telepathy":     {"ally_immune": True, "category": "immunity"},
    "soundproof":    {"sound_immune": True, "category": "immunity"},
    "bulletproof":   {"bullet_immune": True, "category": "immunity"},

    # --- Defensive abilities ---
    "intimidate":    {"on_switch_in_atk_drop": 1, "category": "defensive"},
    "multiscale":    {"full_hp_damage_mult": 0.5, "category": "defensive"},
    "shadowshield":  {"full_hp_damage_mult": 0.5, "category": "defensive"},
    "filter":        {"se_damage_resist": 0.75, "category": "defensive"},
    "solidrock":     {"se_damage_resist": 0.75, "category": "defensive"},
    "prismarmor":    {"se_damage_resist": 0.75, "category": "defensive"},
    "icescales":     {"special_damage_mult": 0.5, "category": "defensive"},
    "furcoat":       {"phys_def_mult": 2.0, "category": "defensive"},
    "marvelscale":   {"status_def_mult": 1.5, "category": "defensive"},
    "unaware":       {"ignores_boosts": True, "category": "defensive"},
    "clearbody":     {"stat_drop_immune": True, "category": "defensive"},
    "whitesmoke":    {"stat_drop_immune": True, "category": "defensive"},

    # --- Recovery abilities ---
    "regenerator":   {"on_switch_out_heal": 1/3, "category": "recovery"},
    "naturalcure":   {"on_switch_out_cure": True, "category": "recovery"},
    "poisonheal":    {"poison_heals": 1/8, "category": "recovery"},
    "raindish":      {"rain_heal": 1/16, "category": "recovery"},
    "icebody":       {"snow_heal": 1/16, "category": "recovery"},

    # --- Weather-setting abilities ---
    "drought":       {"weather": "sun", "category": "weather"},
    "drizzle":       {"weather": "rain", "category": "weather"},
    "sandstream":    {"weather": "sand", "category": "weather"},
    "snowwarning":   {"weather": "snow", "category": "weather"},
    "orichalcumpulse": {"weather": "sun", "category": "weather"},
    "desolateland":  {"weather": "sun", "primal": True, "category": "weather"},
    "primordialsea": {"weather": "rain", "primal": True, "category": "weather"},
    "deltastream":   {"weather": "strongwind", "primal": True, "category": "weather"},

    # --- Terrain-setting abilities ---
    "electricsurge": {"terrain": "electric", "category": "terrain"},
    "grassysurge":   {"terrain": "grassy", "category": "terrain"},
    "psychicsurge":  {"terrain": "psychic", "category": "terrain"},
    "mistysurge":    {"terrain": "misty", "category": "terrain"},
    "hadronengine":  {"terrain": "electric", "electric_spa_mult": 1.33, "category": "terrain"},
    "seedsower":     {"on_hit_terrain": "grassy", "category": "terrain"},

    # --- Speed abilities ---
    "speedboost":    {"end_turn_spe_boost": 1, "category": "speed"},
    "swiftswim":     {"weather_speed": "rain", "spe_mult": 2.0, "category": "speed"},
    "chlorophyll":   {"weather_speed": "sun", "spe_mult": 2.0, "category": "speed"},
    "sandrush":      {"weather_speed": "sand", "spe_mult": 2.0, "category": "speed"},
    "slushrush":     {"weather_speed": "snow", "spe_mult": 2.0, "category": "speed"},
    "surgesurfer":   {"terrain_speed": "electric", "spe_mult": 2.0, "category": "speed"},
    "unburden":      {"on_item_loss_spe_mult": 2.0, "category": "speed"},
    "quickfeet":     {"status_spe_mult": 1.5, "category": "speed"},

    # --- Paradox abilities ---
    "protosynthesis": {"weather_or_booster": "sun", "boosts_highest": True, "boost_mult": 1.3, "category": "paradox"},
    "quarkdrive":     {"terrain_or_booster": "electric", "boosts_highest": True, "boost_mult": 1.3, "category": "paradox"},

    # --- Hazard immunity ---
    "magicguard":    {"indirect_damage_immune": True, "category": "hazard_immune"},
    "magicbounce":   {"reflects_status": True, "category": "hazard_immune"},

    # --- Status immunity ---
    "goodasgold":    {"status_immune": True, "category": "status_immunity"},
    "comatose":      {"status_immune": True, "always_asleep": True, "category": "status_immunity"},
    "purifyingsalt": {"status_immune": True, "ghost_resist": 0.5, "category": "status_immunity"},
    "thermalexchange": {"burn_immune": True, "fire_hit_atk_boost": 1, "category": "status_immunity"},
    "waterveil":     {"burn_immune": True, "category": "status_immunity"},
    "immunity":      {"poison_immune": True, "category": "status_immunity"},
    "pastelveil":    {"poison_immune": True, "ally_too": True, "category": "status_immunity"},
    "oblivious":     {"taunt_immune": True, "infatuation_immune": True, "intimidate_immune": True, "category": "status_immunity"},
    "owntempo":      {"confusion_immune": True, "intimidate_immune": True, "category": "status_immunity"},
    "innerfocus":    {"flinch_immune": True, "intimidate_immune": True, "category": "status_immunity"},
    "limber":        {"paralysis_immune": True, "category": "status_immunity"},
    "insomnia":      {"sleep_immune": True, "category": "status_immunity"},
    "vitalspirit":   {"sleep_immune": True, "category": "status_immunity"},

    # --- Priority abilities ---
    "prankster":     {"status_priority": 1, "category": "priority"},
    "galewings":     {"flying_priority": 1, "full_hp_only": True, "category": "priority"},
    "triage":        {"heal_priority": 3, "category": "priority"},

    # --- Contact punish abilities ---
    "roughskin":     {"contact_recoil": 1/8, "category": "contact_punish"},
    "ironbarbs":     {"contact_recoil": 1/8, "category": "contact_punish"},
    "flamebody":     {"contact_burn_chance": 0.3, "category": "contact_punish"},
    "poisonpoint":   {"contact_poison_chance": 0.3, "category": "contact_punish"},
    "effectspore":   {"contact_status_chance": 0.3, "category": "contact_punish"},
    "gooey":         {"contact_speed_drop": 1, "category": "contact_punish"},
    "tanglinghair":  {"contact_speed_drop": 1, "category": "contact_punish"},
    "cottondown":    {"contact_speed_drop": 1, "category": "contact_punish"},
    "static":        {"contact_para_chance": 0.3, "category": "contact_punish"},
    "cutechar":      {"contact_infatuation_chance": 0.3, "category": "contact_punish"},

    # --- Ability suppression ---
    "moldbreaker":   {"ignores_abilities": True, "category": "ability_break"},
    "teravolt":      {"ignores_abilities": True, "category": "ability_break"},
    "turboblaze":    {"ignores_abilities": True, "category": "ability_break"},
    "myceliummight": {"status_ignores_abilities": True, "category": "ability_break"},

    # --- Boost abilities ---
    "moxie":         {"on_ko_boost": {"atk": 1}, "category": "boost"},
    "beastboost":    {"on_ko_boost_highest": True, "category": "boost"},
    "intrepidsword": {"on_switch_boost": {"atk": 1}, "category": "boost"},
    "dauntlessshield": {"on_switch_boost": {"def": 1}, "category": "boost"},
    "download":      {"on_switch_stat_compare": True, "category": "boost"},

    # --- Aura abilities (field-wide) ---
    "fairyaura":     {"type_aura": "Fairy", "aura_mult": 1.33, "category": "aura"},
    "darkaura":      {"type_aura": "Dark", "aura_mult": 1.33, "category": "aura"},
    "aurabreak":     {"aura_invert": True, "category": "aura"},
}


# ===================================================================
# MEGA STONES REGISTRY (Gen 6-7)
# ===================================================================
# Maps mega stone item ID -> species info. 46 Mega Stones + 2 Primal
# Reversion orbs = 48 total. Species IDs use Showdown format.
# Cross-referenced with: Showdown items.ts megaStone field, Bulbapedia
# ===================================================================

MEGA_STONES: dict[str, dict] = {
    # Gen 6 — Kanto starters + original 28
    "venusaurite":       {"species": "venusaur", "mega_species": "venusaurmega", "category": "mega_stone"},
    "charizarditex":     {"species": "charizard", "mega_species": "charizardmegax", "category": "mega_stone"},
    "charizarditey":     {"species": "charizard", "mega_species": "charizardmegay", "category": "mega_stone"},
    "blastoisinite":     {"species": "blastoise", "mega_species": "blastoisemega", "category": "mega_stone"},
    "alakazite":         {"species": "alakazam", "mega_species": "alakazammega", "category": "mega_stone"},
    "gengarite":         {"species": "gengar", "mega_species": "gengarmega", "category": "mega_stone"},
    "kangaskhanite":     {"species": "kangaskhan", "mega_species": "kangaskhanmega", "category": "mega_stone"},
    "pinsirite":         {"species": "pinsir", "mega_species": "pinsirmega", "category": "mega_stone"},
    "gyaradosite":       {"species": "gyarados", "mega_species": "gyaradosmega", "category": "mega_stone"},
    "aerodactylite":     {"species": "aerodactyl", "mega_species": "aerodactylmega", "category": "mega_stone"},
    "mewtwonitex":       {"species": "mewtwo", "mega_species": "mewtwomegax", "category": "mega_stone"},
    "mewtwonitey":       {"species": "mewtwo", "mega_species": "mewtwomegay", "category": "mega_stone"},
    "ampharosite":       {"species": "ampharos", "mega_species": "ampharosmega", "category": "mega_stone"},
    "scizorite":         {"species": "scizor", "mega_species": "scizormega", "category": "mega_stone"},
    "heracronite":       {"species": "heracross", "mega_species": "heracrossmega", "category": "mega_stone"},
    "houndoominite":     {"species": "houndoom", "mega_species": "houndoommega", "category": "mega_stone"},
    "tyranitarite":      {"species": "tyranitar", "mega_species": "tyranitarmega", "category": "mega_stone"},
    "blazikenite":       {"species": "blaziken", "mega_species": "blazikenmega", "category": "mega_stone"},
    "gardevoirite":      {"species": "gardevoir", "mega_species": "gardevoirmega", "category": "mega_stone"},
    "mawilite":          {"species": "mawile", "mega_species": "mawilemega", "category": "mega_stone"},
    "aggronite":         {"species": "aggron", "mega_species": "aggronmega", "category": "mega_stone"},
    "medichamite":       {"species": "medicham", "mega_species": "medichammega", "category": "mega_stone"},
    "manectite":         {"species": "manectric", "mega_species": "manectricmega", "category": "mega_stone"},
    "banettite":         {"species": "banette", "mega_species": "banettemega", "category": "mega_stone"},
    "absolite":          {"species": "absol", "mega_species": "absolmega", "category": "mega_stone"},
    "garchompite":       {"species": "garchomp", "mega_species": "garchompmega", "category": "mega_stone"},
    "lucarionite":       {"species": "lucario", "mega_species": "lucariomega", "category": "mega_stone"},
    "abomasite":         {"species": "abomasnow", "mega_species": "abomasnowmega", "category": "mega_stone"},
    # Gen 6 ORAS — 19 additional
    "latiasite":         {"species": "latias", "mega_species": "latiasmega", "category": "mega_stone"},
    "latiosite":         {"species": "latios", "mega_species": "latiosmega", "category": "mega_stone"},
    "swampertite":       {"species": "swampert", "mega_species": "swampertmega", "category": "mega_stone"},
    "sceptilite":        {"species": "sceptile", "mega_species": "sceptilemega", "category": "mega_stone"},
    "sablenite":         {"species": "sableye", "mega_species": "sableyemega", "category": "mega_stone"},
    "altarianite":       {"species": "altaria", "mega_species": "altariamega", "category": "mega_stone"},
    "galladite":         {"species": "gallade", "mega_species": "gallademega", "category": "mega_stone"},
    "audinite":          {"species": "audino", "mega_species": "audinomega", "category": "mega_stone"},
    "metagrossite":      {"species": "metagross", "mega_species": "metagrossmega", "category": "mega_stone"},
    "sharpedonite":      {"species": "sharpedo", "mega_species": "sharpedomega", "category": "mega_stone"},
    "slowbronite":       {"species": "slowbro", "mega_species": "slowbromega", "category": "mega_stone"},
    "steelixite":        {"species": "steelix", "mega_species": "steelixmega", "category": "mega_stone"},
    "pidgeotite":        {"species": "pidgeot", "mega_species": "pidgeotmega", "category": "mega_stone"},
    "glalitite":         {"species": "glalie", "mega_species": "glaliemega", "category": "mega_stone"},
    "diancite":          {"species": "diancie", "mega_species": "dianciemega", "category": "mega_stone"},
    "cameruptite":       {"species": "camerupt", "mega_species": "cameruptmega", "category": "mega_stone"},
    "lopunnite":         {"species": "lopunny", "mega_species": "lopunnymega", "category": "mega_stone"},
    "salamencite":       {"species": "salamence", "mega_species": "salamencemega", "category": "mega_stone"},
    "beedrillite":       {"species": "beedrill", "mega_species": "beedrillmega", "category": "mega_stone"},
    # Primal Reversion orbs (functionally similar to Mega Stones)
    "redorb":            {"species": "groudon", "mega_species": "groudonprimal", "category": "mega_stone"},
    "blueorb":           {"species": "kyogre", "mega_species": "kyogreprimal", "category": "mega_stone"},
}


# ===================================================================
# Z-CRYSTALS REGISTRY (Gen 7)
# ===================================================================
# Maps Z-Crystal item ID -> type and Z-Move info.
# 18 type-specific + 15 species-specific = 33 total.
# Species-specific crystals include the exclusive Z-Move name and BP.
# Cross-referenced with: Showdown items.ts zMove/zMoveType/zMoveFrom
# ===================================================================

Z_CRYSTALS: dict[str, dict] = {
    # Type Z-Crystals (18 types)
    "normaliumz":      {"type": "Normal", "z_power": True, "category": "z_crystal"},
    "firiumz":         {"type": "Fire", "z_power": True, "category": "z_crystal"},
    "wateriumz":       {"type": "Water", "z_power": True, "category": "z_crystal"},
    "electriumz":      {"type": "Electric", "z_power": True, "category": "z_crystal"},
    "grassiumz":       {"type": "Grass", "z_power": True, "category": "z_crystal"},
    "iciumz":          {"type": "Ice", "z_power": True, "category": "z_crystal"},
    "fightiniumz":     {"type": "Fighting", "z_power": True, "category": "z_crystal"},
    "poisoniumz":      {"type": "Poison", "z_power": True, "category": "z_crystal"},
    "groundiumz":      {"type": "Ground", "z_power": True, "category": "z_crystal"},
    "flyiniumz":       {"type": "Flying", "z_power": True, "category": "z_crystal"},
    "psychiumz":       {"type": "Psychic", "z_power": True, "category": "z_crystal"},
    "buginiumz":       {"type": "Bug", "z_power": True, "category": "z_crystal"},
    "rockiumz":        {"type": "Rock", "z_power": True, "category": "z_crystal"},
    "ghostiumz":       {"type": "Ghost", "z_power": True, "category": "z_crystal"},
    "dragoniumz":      {"type": "Dragon", "z_power": True, "category": "z_crystal"},
    "darkiniumz":      {"type": "Dark", "z_power": True, "category": "z_crystal"},
    "steeliumz":       {"type": "Steel", "z_power": True, "category": "z_crystal"},
    "fairiumz":        {"type": "Fairy", "z_power": True, "category": "z_crystal"},
    # Species-specific Z-Crystals (15)
    "pikaniumz":       {"type": "Electric", "species": "pikachu", "z_move": "catastropika", "z_base_move": "volttackle", "z_power": 210, "category": "z_crystal"},
    "aloraichiumz":    {"type": "Electric", "species": "raichualola", "z_move": "stokedsparksurfer", "z_base_move": "thunderbolt", "z_power": 175, "category": "z_crystal"},
    "decidiumz":       {"type": "Ghost", "species": "decidueye", "z_move": "sinisterarrowraid", "z_base_move": "spiritshackle", "z_power": 180, "category": "z_crystal"},
    "inciniumz":       {"type": "Dark", "species": "incineroar", "z_move": "maliciousmoonsault", "z_base_move": "darkestlariat", "z_power": 180, "category": "z_crystal"},
    "primariumz":      {"type": "Water", "species": "primarina", "z_move": "oceanicoperetta", "z_base_move": "sparklingaria", "z_power": 195, "category": "z_crystal"},
    "tapuniumz":       {"type": "Fairy", "species": "tapukoko,tapulele,tapubulu,tapufini", "z_move": "guardianofalola", "z_base_move": "naturesmadness", "z_power": 0, "category": "z_crystal"},
    "marshadiumz":     {"type": "Ghost", "species": "marshadow", "z_move": "soulstealing7starstrike", "z_base_move": "spectralthief", "z_power": 195, "category": "z_crystal"},
    "snorliumz":       {"type": "Normal", "species": "snorlax", "z_move": "pulverizingpancake", "z_base_move": "gigaimpact", "z_power": 210, "category": "z_crystal"},
    "eeviumz":         {"type": "Normal", "species": "eevee", "z_move": "extremeevoboost", "z_base_move": "lastresort", "z_power": 0, "category": "z_crystal"},
    "mewniumz":        {"type": "Psychic", "species": "mew", "z_move": "genesissupernova", "z_base_move": "psychic", "z_power": 185, "category": "z_crystal"},
    "pikashuniumz":    {"type": "Electric", "species": "pikachuoriginal,pikachuhoenn,pikachusinnoh,pikachuunova,pikachukalos,pikachualola,pikachupartner", "z_move": "10000000voltthunderbolt", "z_base_move": "thunderbolt", "z_power": 195, "category": "z_crystal"},
    "lycaniumz":       {"type": "Rock", "species": "lycanroc,lycanrocmidnight,lycanrocdusk", "z_move": "splinteredstormshards", "z_base_move": "stoneedge", "z_power": 190, "category": "z_crystal"},
    "mimikiumz":       {"type": "Fairy", "species": "mimikyu", "z_move": "letssnuggleforever", "z_base_move": "playrough", "z_power": 190, "category": "z_crystal"},
    "kommoniumz":      {"type": "Dragon", "species": "kommoo", "z_move": "clangoroussoulblaze", "z_base_move": "clangingscales", "z_power": 185, "category": "z_crystal"},
    "solganiumz":      {"type": "Steel", "species": "solgaleo,necrozmaduskmane", "z_move": "searingsunrazesmash", "z_base_move": "sunsteelstrike", "z_power": 200, "category": "z_crystal"},
    "lunaliumz":       {"type": "Ghost", "species": "lunala,necrozmadawnwings", "z_move": "menacingmoonrazemaelstrom", "z_base_move": "moongeistbeam", "z_power": 200, "category": "z_crystal"},
    "ultranecroziumz": {"type": "Psychic", "species": "necrozmaultra", "z_move": "lightthatburnsthesky", "z_base_move": "photongeyser", "z_power": 200, "category": "z_crystal"},
}


# ===================================================================
# Z-MOVE BASE POWER TABLE (Gen 7)
# ===================================================================
# Converts a normal move's base power to its Z-Move base power.
# Source: Bulbapedia Z-Move article, Showdown source (data/moves.ts)
# ===================================================================

def get_z_move_bp(original_bp: int) -> int:
    """Convert a normal move's base power to its Z-Move base power.

    Args:
        original_bp: The base power of the original move.

    Returns:
        The base power of the corresponding Z-Move.
    """
    if original_bp <= 55:
        return 100
    if original_bp <= 65:
        return 120
    if original_bp <= 75:
        return 140
    if original_bp <= 85:
        return 160
    if original_bp <= 95:
        return 175
    if original_bp <= 100:
        return 180
    if original_bp <= 110:
        return 185
    if original_bp <= 125:
        return 190
    if original_bp <= 130:
        return 195
    return 200


# ===================================================================
# MAX MOVE BASE POWER TABLE (Gen 8 — Dynamax)
# ===================================================================
# Converts a normal move's base power to its Max Move base power.
# Fighting and Poison types have lower scaling (Max Knuckle/Ooze).
# Source: Bulbapedia Dynamax article, Showdown source
# ===================================================================

def get_max_move_bp(original_bp: int, move_type: str = "") -> int:
    """Convert a normal move's base power to its Max Move base power.

    Fighting and Poison type moves have reduced Max Move BP because
    Max Knuckle and Max Ooze provide team-wide stat boosts.

    Args:
        original_bp: The base power of the original move.
        move_type: The type of the original move (e.g., "Fighting").

    Returns:
        The base power of the corresponding Max Move.
    """
    reduced = move_type in ("Fighting", "Poison")
    if original_bp <= 40:
        return 70 if reduced else 90
    if original_bp <= 50:
        return 75 if reduced else 100
    if original_bp <= 60:
        return 80 if reduced else 110
    if original_bp <= 70:
        return 85 if reduced else 120
    if original_bp <= 100:
        return 90 if reduced else 130
    if original_bp <= 140:
        return 95 if reduced else 140
    return 100 if reduced else 150


# ===================================================================
# GIMMICK MECHANIC HELPERS
# ===================================================================

def is_mega_stone(item_id: str) -> bool:
    """Check if the given item ID is a Mega Stone (or Primal Orb)."""
    return item_id in MEGA_STONES


def is_z_crystal(item_id: str) -> bool:
    """Check if the given item ID is a Z-Crystal."""
    return item_id in Z_CRYSTALS


def get_mega_species(item_id: str) -> str | None:
    """Get the mega-evolved species ID for a given Mega Stone.

    Args:
        item_id: The Showdown item ID (e.g., "charizarditex").

    Returns:
        The mega species ID (e.g., "charizardmegax"), or None if not a Mega Stone.
    """
    entry = MEGA_STONES.get(item_id)
    return entry["mega_species"] if entry else None


def get_z_crystal_type(item_id: str) -> str | None:
    """Get the type associated with a Z-Crystal.

    Args:
        item_id: The Showdown item ID (e.g., "firiumz").

    Returns:
        The type string (e.g., "Fire"), or None if not a Z-Crystal.
    """
    entry = Z_CRYSTALS.get(item_id)
    return entry["type"] if entry else None


def get_z_move_for_species(item_id: str) -> str | None:
    """Get the exclusive Z-Move name for a species-specific Z-Crystal.

    Args:
        item_id: The Showdown item ID (e.g., "pikaniumz").

    Returns:
        The Z-Move ID (e.g., "catastropika"), or None if not species-specific.
    """
    entry = Z_CRYSTALS.get(item_id)
    if entry and "z_move" in entry:
        return entry["z_move"]
    return None


def get_gimmick_gen(item_id: str) -> int | None:
    """Get the generation a gimmick item belongs to.

    Returns:
        6 or 7 for Mega Stones, 7 for Z-Crystals, or None.
    """
    if item_id in MEGA_STONES:
        # Primal orbs and most megas are Gen 6, ORAS additions also Gen 6
        return 6
    if item_id in Z_CRYSTALS:
        return 7
    return None


# ===================================================================
# MOVE CATEGORIZATION FUNCTIONS
# ===================================================================
# These build move category sets from actual moves.json data,
# replacing hardcoded name sets.
# ===================================================================

def build_move_sets(moves_data: dict[str, dict]) -> dict[str, set[str]]:
    """Build all move category sets from Showdown moves.json data.

    Args:
        moves_data: Raw moves dict from PokemonDataLoader.moves

    Returns:
        Dict mapping category name to set of move IDs.
    """
    sets: dict[str, set[str]] = {
        "priority": set(),
        "drain": set(),
        "recoil": set(),
        "spread": set(),
        "setup": set(),
        "recovery": set(),
        "pivot": set(),
        "hazard": set(),
        "removal": set(),
        "phazing": set(),
        "status_brn": set(),
        "status_par": set(),
        "status_slp": set(),
        "status_tox": set(),
        "status_psn": set(),
        "status_frz": set(),
        "contact": set(),
        "sound": set(),
        "punch": set(),
        "bite": set(),
        "pulse": set(),
        "slicing": set(),
        "wind": set(),
        "bullet": set(),
    }

    for mid, mdata in moves_data.items():
        # Skip non-standard moves
        if mdata.get("isNonstandard") == "Past":
            continue

        flags = mdata.get("flags", {})

        # Priority moves (priority > 0)
        if mdata.get("priority", 0) > 0:
            sets["priority"].add(mid)

        # Drain moves (move.drain exists)
        if mdata.get("drain"):
            sets["drain"].add(mid)

        # Recoil moves (move.recoil exists)
        if mdata.get("recoil"):
            sets["recoil"].add(mid)

        # Spread moves (multi-target)
        target = mdata.get("target", "")
        if target in ("allAdjacentFoes", "allAdjacent", "all"):
            sets["spread"].add(mid)

        # Setup moves (self-targeting boosts)
        boosts = mdata.get("boosts")
        if boosts and mdata.get("target") == "self":
            if any(v > 0 for v in boosts.values()):
                sets["setup"].add(mid)

        # Recovery moves (flags.heal OR is a self-healing status move)
        if flags.get("heal"):
            sets["recovery"].add(mid)
        # Also add direct HP recovery moves (heal: [n, d])
        if mdata.get("heal") and mdata.get("target") == "self":
            sets["recovery"].add(mid)

        # Pivot moves (selfSwitch is truthy)
        if mdata.get("selfSwitch"):
            sets["pivot"].add(mid)

        # Hazard moves (sideCondition set)
        if mdata.get("sideCondition"):
            sets["hazard"].add(mid)

        # Phazing moves (forceSwitch)
        if mdata.get("forceSwitch"):
            sets["phazing"].add(mid)

        # Flag-based categories
        if flags.get("contact"):
            sets["contact"].add(mid)
        if flags.get("sound"):
            sets["sound"].add(mid)
        if flags.get("punch"):
            sets["punch"].add(mid)
        if flags.get("bite"):
            sets["bite"].add(mid)
        if flags.get("pulse"):
            sets["pulse"].add(mid)
        if flags.get("slicing"):
            sets["slicing"].add(mid)
        if flags.get("wind"):
            sets["wind"].add(mid)
        if flags.get("bullet"):
            sets["bullet"].add(mid)

        # Status-inflicting moves: check primary status, secondary, and secondaries
        _extract_statuses(mid, mdata, sets)

    # Hazard removal moves (hard to detect from JSON, use known set + defog check)
    sets["removal"] = {"rapidspin", "defog", "tidyup", "courtchange", "mortalspinx"}

    log.info(
        "Built move sets: priority=%d, drain=%d, recoil=%d, spread=%d, "
        "setup=%d, recovery=%d, pivot=%d, hazard=%d, phazing=%d, contact=%d",
        len(sets["priority"]), len(sets["drain"]), len(sets["recoil"]),
        len(sets["spread"]), len(sets["setup"]), len(sets["recovery"]),
        len(sets["pivot"]), len(sets["hazard"]), len(sets["phazing"]),
        len(sets["contact"]),
    )

    return sets


def _extract_statuses(mid: str, mdata: dict, sets: dict[str, set[str]]) -> None:
    """Extract status-inflicting move IDs from a move's primary + secondary effects."""
    status_map = {
        "brn": "status_brn",
        "par": "status_par",
        "slp": "status_slp",
        "tox": "status_tox",
        "psn": "status_psn",
        "frz": "status_frz",
    }

    # Primary status effect (e.g., Thunder Wave -> status: "par")
    primary_status = mdata.get("status")
    if primary_status and primary_status in status_map:
        sets[status_map[primary_status]].add(mid)

    # Single secondary
    secondary = mdata.get("secondary")
    if secondary and isinstance(secondary, dict):
        sec_status = secondary.get("status")
        if sec_status and sec_status in status_map:
            sets[status_map[sec_status]].add(mid)

    # Multiple secondaries (e.g., Fire Fang: burn + flinch)
    for sec in mdata.get("secondaries", []):
        if isinstance(sec, dict):
            sec_status = sec.get("status")
            if sec_status and sec_status in status_map:
                sets[status_map[sec_status]].add(mid)


def get_move_secondary_chance(mdata: dict) -> float:
    """Get the highest secondary effect chance for a move (0-100)."""
    chances = []
    secondary = mdata.get("secondary")
    if secondary and isinstance(secondary, dict):
        c = secondary.get("chance", 0)
        if c:
            chances.append(c)
    for sec in mdata.get("secondaries", []):
        if isinstance(sec, dict):
            c = sec.get("chance", 0)
            if c:
                chances.append(c)
    return max(chances) if chances else 0


def get_move_drain_fraction(mdata: dict) -> float:
    """Get drain fraction for a move (0 to 1). E.g., Drain Punch = 0.5."""
    drain = mdata.get("drain")
    if drain and len(drain) == 2 and drain[1] > 0:
        return drain[0] / drain[1]
    return 0.0


def get_move_recoil_fraction(mdata: dict) -> float:
    """Get recoil fraction for a move (0 to 1). E.g., Flare Blitz = 0.33."""
    recoil = mdata.get("recoil")
    if recoil and len(recoil) == 2 and recoil[1] > 0:
        return recoil[0] / recoil[1]
    return 0.0


def get_move_self_stat_drops(mdata: dict) -> dict[str, int]:
    """Get self-inflicted stat changes from a move (e.g., Close Combat: {def: -1, spd: -1})."""
    self_effect = mdata.get("self")
    if self_effect and isinstance(self_effect, dict):
        boosts = self_effect.get("boosts")
        if boosts:
            return boosts
    return {}


def get_move_boost_total(mdata: dict) -> int:
    """Get total stat boost stages from a setup move (e.g., Dragon Dance = 2)."""
    boosts = mdata.get("boosts")
    if boosts and mdata.get("target") == "self":
        return sum(max(v, 0) for v in boosts.values())
    return 0


# ===================================================================
# DAMAGE MODIFIER HELPERS
# ===================================================================

def get_item_atk_mult(item_id: str, is_physical: bool) -> float:
    """Get attack stat multiplier from item."""
    eff = ITEM_EFFECTS.get(item_id, {})
    if is_physical:
        return eff.get("atk_mult", 1.0)
    else:
        return eff.get("spa_mult", 1.0)


def get_item_def_mult(item_id: str, is_physical: bool) -> float:
    """Get defense stat multiplier from item."""
    eff = ITEM_EFFECTS.get(item_id, {})
    if is_physical:
        return eff.get("def_mult", 1.0)
    else:
        return eff.get("spd_mult", 1.0)


def get_item_damage_mult(item_id: str) -> float:
    """Get final damage multiplier from item (e.g., Life Orb 1.3x)."""
    return ITEM_EFFECTS.get(item_id, {}).get("damage_mult", 1.0)


def get_item_spe_mult(item_id: str) -> float:
    """Get speed multiplier from item (e.g., Choice Scarf 1.5x)."""
    return ITEM_EFFECTS.get(item_id, {}).get("spe_mult", 1.0)


def get_ability_atk_mult(ability_id: str) -> float:
    """Get attack stat multiplier from ability (e.g., Huge Power 2.0x)."""
    return ABILITY_EFFECTS.get(ability_id, {}).get("atk_mult", 1.0)


def get_ability_stab_mult(ability_id: str) -> float:
    """Get STAB multiplier override (default 1.5, Adaptability = 2.0)."""
    return ABILITY_EFFECTS.get(ability_id, {}).get("stab_mult", 1.5)


def get_ability_type_immunity(ability_id: str) -> str | None:
    """Get type that this ability grants immunity to (e.g., Levitate -> 'Ground')."""
    return ABILITY_EFFECTS.get(ability_id, {}).get("type_immunity")


def get_ability_bp_modifier(ability_id: str, move_flags: dict, bp: int, has_secondary: bool) -> float:
    """Get base power modifier from ability based on move flags and BP.

    Returns multiplier to apply to base power.
    """
    eff = ABILITY_EFFECTS.get(ability_id, {})
    mult = 1.0

    # Technician: 1.5x for moves with BP <= 60
    if eff.get("bp_threshold") and bp <= eff["bp_threshold"]:
        mult *= eff.get("low_bp_mult", 1.0)

    # Flag-based multipliers
    if eff.get("contact_mult") and move_flags.get("contact"):
        mult *= eff["contact_mult"]
    if eff.get("punch_mult") and move_flags.get("punch"):
        mult *= eff["punch_mult"]
    if eff.get("bite_mult") and move_flags.get("bite"):
        mult *= eff["bite_mult"]
    if eff.get("pulse_mult") and move_flags.get("pulse"):
        mult *= eff["pulse_mult"]
    if eff.get("sound_mult") and move_flags.get("sound"):
        mult *= eff["sound_mult"]

    # Sheer Force: 1.3x if move has secondary effect
    if eff.get("no_secondary_mult") and has_secondary:
        mult *= eff["no_secondary_mult"]

    return mult


def get_item_bp_modifier(item_id: str, move_flags: dict, is_physical: bool) -> float:
    """Get base power modifier from item based on move flags."""
    eff = ITEM_EFFECTS.get(item_id, {})
    mult = 1.0

    if eff.get("phys_damage_mult") and is_physical:
        mult *= eff["phys_damage_mult"]
    if eff.get("spec_damage_mult") and not is_physical:
        mult *= eff["spec_damage_mult"]
    if eff.get("punch_damage_mult") and move_flags.get("punch"):
        mult *= eff["punch_damage_mult"]

    return mult


# ===================================================================
# CATEGORY QUERY FUNCTIONS
# ===================================================================

def is_choice_item(item_id: str) -> bool:
    return ITEM_EFFECTS.get(item_id, {}).get("category") == "choice"


def is_offensive_item(item_id: str) -> bool:
    cat = ITEM_EFFECTS.get(item_id, {}).get("category", "")
    return cat in ("offensive", "choice", "type_boost")


def is_defensive_item(item_id: str) -> bool:
    return ITEM_EFFECTS.get(item_id, {}).get("category") == "defensive"


def is_recovery_item(item_id: str) -> bool:
    return ITEM_EFFECTS.get(item_id, {}).get("category") == "recovery"


def is_berry(item_id: str) -> bool:
    eff = ITEM_EFFECTS.get(item_id, {})
    return eff.get("is_berry", False) or eff.get("category") in ("berry", "resist_berry")


def is_hazard_immune_item(item_id: str) -> bool:
    return ITEM_EFFECTS.get(item_id, {}).get("hazard_immune", False)


def ability_has_type_immunity(ability_id: str) -> bool:
    return "type_immunity" in ABILITY_EFFECTS.get(ability_id, {})


def ability_sets_weather(ability_id: str) -> bool:
    return "weather" in ABILITY_EFFECTS.get(ability_id, {})


def ability_sets_terrain(ability_id: str) -> bool:
    return "terrain" in ABILITY_EFFECTS.get(ability_id, {})


def ability_is_intimidate(ability_id: str) -> bool:
    return "on_switch_in_atk_drop" in ABILITY_EFFECTS.get(ability_id, {})


def ability_is_regenerator(ability_id: str) -> bool:
    return "on_switch_out_heal" in ABILITY_EFFECTS.get(ability_id, {})


def ability_boosts_power(ability_id: str) -> bool:
    cat = ABILITY_EFFECTS.get(ability_id, {}).get("category", "")
    return cat in ("power_boost", "type_change")


def ability_boosts_speed(ability_id: str) -> bool:
    cat = ABILITY_EFFECTS.get(ability_id, {}).get("category", "")
    return cat in ("speed", "paradox")


def ability_ignores_abilities(ability_id: str) -> bool:
    return ABILITY_EFFECTS.get(ability_id, {}).get("ignores_abilities", False)


def ability_punishes_contact(ability_id: str) -> bool:
    return ABILITY_EFFECTS.get(ability_id, {}).get("category") == "contact_punish"


def ability_immune_to_status(ability_id: str) -> bool:
    cat = ABILITY_EFFECTS.get(ability_id, {}).get("category", "")
    return cat == "status_immunity" or ABILITY_EFFECTS.get(ability_id, {}).get("status_immune", False)


# ===================================================================
# VERIFICATION
# ===================================================================

def verify_mechanics(pokemon_data) -> dict[str, int]:
    """Verify mechanics registries against loaded Showdown data.

    Returns dict with counts of verified/missing/extra entries.
    """
    results = {"items_verified": 0, "items_missing": 0,
               "abilities_verified": 0, "abilities_missing": 0}

    if pokemon_data and pokemon_data.items:
        for item_id in ITEM_EFFECTS:
            if item_id in pokemon_data.items:
                results["items_verified"] += 1
            else:
                results["items_missing"] += 1
                log.warning("Item in registry but not in Showdown data: %s", item_id)

    if pokemon_data and pokemon_data.abilities:
        for ability_id in ABILITY_EFFECTS:
            clean_id = ability_id.replace(" ", "")
            if clean_id in pokemon_data.abilities or ability_id in pokemon_data.abilities:
                results["abilities_verified"] += 1
            else:
                results["abilities_missing"] += 1
                log.warning("Ability in registry but not in Showdown data: %s", ability_id)

    log.info(
        "Mechanics verification: items %d/%d, abilities %d/%d",
        results["items_verified"], len(ITEM_EFFECTS),
        results["abilities_verified"], len(ABILITY_EFFECTS),
    )
    return results
