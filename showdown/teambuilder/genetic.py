"""Genetic algorithm team builder.

Evolves Pokemon teams by:
1. Initializing a population of random legal teams
2. Evaluating each team's predicted win rate against the meta
3. Selecting top performers, crossing over, and mutating
4. Repeating until convergence
"""

import copy
import logging
import random
import re
from typing import Any

from .constraints import FormatConstraints
from .evaluator import TeamEvaluator

log = logging.getLogger("showdown.teambuilder.genetic")


def _to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class GeneticTeamBuilder:
    """Build optimal teams using a genetic algorithm guided by ML predictions."""

    def __init__(
        self,
        evaluator: TeamEvaluator,
        constraints: FormatConstraints,
        pokemon_pool: list[dict],
        population_size: int = 100,
        generations: int = 80,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        elite_size: int = 20,
        tournament_size: int = 5,
        seed: int | None = None,
    ):
        self.evaluator = evaluator
        self.constraints = constraints
        self.pokemon_pool = constraints.filter_legal_pokemon(pokemon_pool)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size

        if seed is not None:
            random.seed(seed)

        # Index pool by species for fast lookup
        self._pool_by_species: dict[str, list[dict]] = {}
        for p in self.pokemon_pool:
            sp = _to_id(p.get("species", ""))
            self._pool_by_species.setdefault(sp, []).append(p)

        self._all_species = list(self._pool_by_species.keys())

        if not self._all_species:
            raise ValueError("Pokemon pool is empty after filtering!")

        log.info(
            "GeneticTeamBuilder initialized: %d unique species, %d total sets in pool",
            len(self._all_species), len(self.pokemon_pool),
        )

    def build(
        self,
        n_results: int = 5,
        progress_callback=None,
    ) -> list[dict[str, Any]]:
        """Run the genetic algorithm and return the top N teams.

        Returns list of dicts with keys:
            - 'team': list of pokemon set dicts
            - 'fitness': predicted win rate
            - 'generation': when this team was found
        """
        population = self._init_population()
        best_ever = []

        for gen in range(1, self.generations + 1):
            # Evaluate
            fitness_scores = [(team, self.evaluator.evaluate(team)) for team in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)

            best_fitness = fitness_scores[0][1]
            avg_fitness = sum(f for _, f in fitness_scores) / len(fitness_scores)

            # Track best
            for team, fitness in fitness_scores[:self.elite_size]:
                key = self.evaluator._team_key(team)
                if not any(self.evaluator._team_key(t["team"]) == key for t in best_ever):
                    best_ever.append({
                        "team": copy.deepcopy(team),
                        "fitness": fitness,
                        "generation": gen,
                    })

            # Prune best_ever to top results
            best_ever.sort(key=lambda x: x["fitness"], reverse=True)
            best_ever = best_ever[:n_results * 3]

            if gen % 10 == 0 or gen == 1:
                log.info(
                    "Gen %d/%d | best=%.4f | avg=%.4f | pool=%d",
                    gen, self.generations, best_fitness, avg_fitness, len(best_ever),
                )

            if progress_callback:
                progress_callback(gen, best_fitness, avg_fitness)

            # Check convergence (top 10 teams within 0.5% of each other)
            if gen > 15 and best_fitness > 0.5:
                top_10_fitness = [f for _, f in fitness_scores[:10]]
                if max(top_10_fitness) - min(top_10_fitness) < 0.005:
                    log.info("Converged at generation %d", gen)
                    break

            # Selection and breeding
            new_population = []

            # Elitism: carry forward top performers
            for team, _ in fitness_scores[:self.elite_size]:
                new_population.append(copy.deepcopy(team))

            # Fill rest with offspring
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1 = self._tournament_select(fitness_scores)
                    parent2 = self._tournament_select(fitness_scores)
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(
                        self._tournament_select(fitness_scores)
                    )

                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                # Ensure legality
                legal, _ = self.constraints.is_team_legal(child)
                if legal:
                    new_population.append(child)
                else:
                    # Try to fix, otherwise skip
                    fixed = self._fix_team(child)
                    if fixed:
                        new_population.append(fixed)

            population = new_population[:self.population_size]

        # Final results — select diverse top teams
        best_ever.sort(key=lambda x: x["fitness"], reverse=True)
        return self._select_diverse_results(best_ever, n_results)

    def _select_diverse_results(
        self, candidates: list[dict], n: int
    ) -> list[dict]:
        """Select top N results maximizing both fitness and diversity."""
        if len(candidates) <= n:
            return candidates

        selected = [candidates[0]]  # Always take the best
        for candidate in candidates[1:]:
            if len(selected) >= n:
                break
            # Check species overlap with all selected teams
            cand_species = {_to_id(p.get("species", "")) for p in candidate["team"]}
            max_overlap = 0
            for sel in selected:
                sel_species = {_to_id(p.get("species", "")) for p in sel["team"]}
                overlap = len(cand_species & sel_species)
                max_overlap = max(max_overlap, overlap)
            # Accept if at least 2 different Pokemon from every selected team
            if max_overlap <= 4:
                selected.append(candidate)

        # If we couldn't find enough diverse teams, fill from remaining
        if len(selected) < n:
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= n:
                    break

        return selected[:n]

    def _init_population(self) -> list[list[dict]]:
        """Create initial random population of legal teams."""
        population = []
        attempts = 0
        max_attempts = self.population_size * 10

        while len(population) < self.population_size and attempts < max_attempts:
            team = self._random_team()
            legal, _ = self.constraints.is_team_legal(team)
            if legal:
                population.append(team)
            attempts += 1

        if len(population) < self.population_size:
            log.warning(
                "Could only generate %d legal teams (target: %d)",
                len(population), self.population_size,
            )

        return population

    def _random_team(self) -> list[dict]:
        """Generate a random 6-Pokemon team from the pool."""
        species = random.sample(
            self._all_species,
            min(6, len(self._all_species)),
        )
        team = []
        for sp in species:
            sets = self._pool_by_species[sp]
            team.append(copy.deepcopy(random.choice(sets)))
        return team

    def _tournament_select(
        self, fitness_scores: list[tuple[list[dict], float]]
    ) -> list[dict]:
        """Select a team via tournament selection."""
        contestants = random.sample(
            fitness_scores,
            min(self.tournament_size, len(fitness_scores)),
        )
        winner = max(contestants, key=lambda x: x[1])
        return winner[0]

    def _crossover(
        self, parent1: list[dict], parent2: list[dict]
    ) -> list[dict]:
        """Create offspring by combining Pokemon from two parent teams."""
        # Uniform crossover: each slot picked from one parent
        child = []
        used_species = set()

        all_pokemon = list(parent1) + list(parent2)
        random.shuffle(all_pokemon)

        for pkmn in all_pokemon:
            sp = _to_id(pkmn.get("species", ""))
            base_sp = self.constraints._base_species(sp)
            if base_sp not in used_species and len(child) < 6:
                child.append(copy.deepcopy(pkmn))
                used_species.add(base_sp)

        # If we don't have 6, fill from pool
        while len(child) < 6:
            sp = random.choice(self._all_species)
            base_sp = self.constraints._base_species(sp)
            if base_sp not in used_species:
                sets = self._pool_by_species[sp]
                child.append(copy.deepcopy(random.choice(sets)))
                used_species.add(base_sp)

        return child[:6]

    def _mutate(self, team: list[dict]) -> list[dict]:
        """Apply random mutation to a team."""
        mutation_type = random.random()

        if mutation_type < 0.5:
            # Replace a random Pokemon with a new one
            team = self._mutate_replace(team)
        elif mutation_type < 0.75:
            # Change a Pokemon's moveset
            team = self._mutate_moveset(team)
        else:
            # Change a Pokemon's item
            team = self._mutate_item(team)

        return team

    def _mutate_replace(self, team: list[dict]) -> list[dict]:
        """Replace one Pokemon on the team with a random one from the pool."""
        team = copy.deepcopy(team)
        if not team:
            return team

        idx = random.randint(0, len(team) - 1)
        used_species = {
            self.constraints._base_species(_to_id(p.get("species", "")))
            for j, p in enumerate(team) if j != idx
        }

        # Find a replacement that doesn't violate species clause
        candidates = [
            sp for sp in self._all_species
            if self.constraints._base_species(sp) not in used_species
        ]
        if candidates:
            new_species = random.choice(candidates)
            sets = self._pool_by_species[new_species]
            team[idx] = copy.deepcopy(random.choice(sets))

        return team

    def _mutate_moveset(self, team: list[dict]) -> list[dict]:
        """Swap to a different set for one Pokemon on the team."""
        team = copy.deepcopy(team)
        if not team:
            return team

        idx = random.randint(0, len(team) - 1)
        sp = _to_id(team[idx].get("species", ""))
        sets = self._pool_by_species.get(sp, [])
        if len(sets) > 1:
            team[idx] = copy.deepcopy(random.choice(sets))

        return team

    def _mutate_item(self, team: list[dict]) -> list[dict]:
        """Change one Pokemon's item to a different one seen in the pool."""
        team = copy.deepcopy(team)
        if not team:
            return team

        idx = random.randint(0, len(team) - 1)
        sp = _to_id(team[idx].get("species", ""))
        sets = self._pool_by_species.get(sp, [])
        items = [s.get("item") for s in sets if s.get("item")]
        if items:
            team[idx]["item"] = random.choice(items)

        return team

    def _fix_team(self, team: list[dict]) -> list[dict] | None:
        """Attempt to fix an illegal team. Returns None if unfixable."""
        # Fix species clause violations
        seen_base = set()
        fixed = []
        for pkmn in team:
            sp = _to_id(pkmn.get("species", ""))
            base = self.constraints._base_species(sp)
            if base not in seen_base:
                fixed.append(pkmn)
                seen_base.add(base)

        # Fill missing slots
        while len(fixed) < 6:
            candidates = [
                sp for sp in self._all_species
                if self.constraints._base_species(sp) not in seen_base
            ]
            if not candidates:
                break
            sp = random.choice(candidates)
            sets = self._pool_by_species[sp]
            fixed.append(copy.deepcopy(random.choice(sets)))
            seen_base.add(self.constraints._base_species(sp))

        legal, _ = self.constraints.is_team_legal(fixed)
        return fixed if legal else None
