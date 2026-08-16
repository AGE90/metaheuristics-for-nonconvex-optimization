from collections.abc import Callable, Sequence

import numpy as np

from metaheuristics.result import OptimizationResult


class GeneticAlgorithm:
    """Binary-encoded genetic algorithm for minimizing a real-valued objective.

    Each decision variable is encoded as a fixed-length bitstring and
    linearly decoded back into its bounded range before the objective is
    evaluated, so the same ``optimize(objective_fn, bounds)`` interface used
    by the other algorithms in this package works here too.

    Parameters
    ----------
    population_size : int
        Number of individuals in the population.
    max_generations : int
        Number of generations to run.
    crossover_rate : float
        Probability that two parents produce offspring via crossover.
    mutation_rate : float
        Per-bit probability of a bit flip.
    elitism : bool
        Whether the best individuals survive unmodified into the next generation.
    elite_size : int
        Number of elite individuals preserved per generation, if ``elitism``.
    bits_per_variable : int
        Bits used to encode each real-valued decision variable.
    tournament_size : int
        Number of individuals competing in each tournament selection.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.01,
        elitism: bool = True,
        elite_size: int = 2,
        bits_per_variable: int = 16,
        tournament_size: int = 3,
    ) -> None:
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.elite_size = elite_size
        self.bits_per_variable = bits_per_variable
        self.tournament_size = tournament_size

    def _decode(self, chromosomes: np.ndarray) -> np.ndarray:
        """Decode a batch of bitstrings into real-valued vectors within ``self.bounds``."""
        bits = self.bits_per_variable
        max_int = 2**bits - 1
        powers = 2 ** np.arange(bits - 1, -1, -1)
        decoded = np.empty((chromosomes.shape[0], len(self.bounds)))
        for i, (low, high) in enumerate(self.bounds):
            segment = chromosomes[:, i * bits : (i + 1) * bits]
            integer_value = segment @ powers
            decoded[:, i] = low + (integer_value / max_int) * (high - low)
        return decoded

    def initialize_population(self) -> np.ndarray:
        """Create a random initial population of bitstrings."""
        gene_length = self.bits_per_variable * len(self.bounds)
        return np.random.randint(2, size=(self.population_size, gene_length))

    def evaluate_population(self, individuals: np.ndarray) -> np.ndarray:
        """Decode and evaluate the objective for each individual (one per row)."""
        decoded = self._decode(individuals)
        return np.array([self.objective_fn(x) for x in decoded])

    def tournament_selection(
        self, population: np.ndarray, fitness_values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select parents by tournament, keeping the lowest (best) fitness in each tournament."""
        num_individuals = population.shape[0]
        selected = np.empty_like(population)
        selected_fitness = np.empty(num_individuals)
        for i in range(num_individuals):
            tournament_indices = np.random.choice(num_individuals, self.tournament_size, replace=False)
            winner_index = tournament_indices[np.argmin(fitness_values[tournament_indices])]
            selected[i, :] = population[winner_index, :]
            selected_fitness[i] = fitness_values[winner_index]
        return selected, selected_fitness

    def single_point_crossover(self, individuals: np.ndarray) -> np.ndarray:
        """Perform single-point crossover on consecutive pairs of individuals."""
        num_individuals, num_genes = individuals.shape
        offspring = np.empty_like(individuals)

        for i in range(0, num_individuals - 1, 2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, num_genes)
                offspring[i, :crossover_point] = individuals[i, :crossover_point]
                offspring[i, crossover_point:] = individuals[i + 1, crossover_point:]
                offspring[i + 1, :crossover_point] = individuals[i + 1, :crossover_point]
                offspring[i + 1, crossover_point:] = individuals[i, crossover_point:]
            else:
                offspring[i, :] = individuals[i, :]
                offspring[i + 1, :] = individuals[i + 1, :]

        if num_individuals % 2:
            offspring[-1, :] = individuals[-1, :]

        return offspring

    def bit_flip_mutation(self, individuals: np.ndarray) -> np.ndarray:
        """Flip each bit independently with probability ``self.mutation_rate``."""
        mask = np.random.rand(*individuals.shape) < self.mutation_rate
        individuals[mask] = 1 - individuals[mask]
        return individuals

    def elitism_selection(
        self, individuals: np.ndarray, fitness_values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``elite_size`` individuals with the lowest (best) fitness."""
        elite_size = min(self.elite_size, individuals.shape[0])
        elite_indices = np.argsort(fitness_values)[:elite_size]
        return individuals[elite_indices, :], fitness_values[elite_indices]

    def optimize(
        self, objective_fn: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]]
    ) -> OptimizationResult:
        """Run the genetic algorithm to minimize ``objective_fn`` over ``bounds``."""
        self.objective_fn = objective_fn
        self.bounds = bounds

        population = self.initialize_population()
        fitness_values = self.evaluate_population(population)

        def best_of(pop, fit):
            idx = np.argmin(fit)
            return self._decode(pop[idx : idx + 1])[0], fit[idx]

        best_position, best_fitness = best_of(population, fitness_values)
        fitness_history, position_history = [best_fitness], [best_position]

        for _ in range(self.max_generations):
            parents, _ = self.tournament_selection(population, fitness_values)
            offspring = self.single_point_crossover(parents)
            offspring = self.bit_flip_mutation(offspring)
            offspring_fitness = self.evaluate_population(offspring)

            if self.elitism:
                elites, elite_fitness = self.elitism_selection(population, fitness_values)
                worst_indices = np.argsort(offspring_fitness)[-len(elites) :]
                offspring[worst_indices] = elites
                offspring_fitness[worst_indices] = elite_fitness

            population, fitness_values = offspring, offspring_fitness
            best_position, best_fitness = best_of(population, fitness_values)
            fitness_history.append(best_fitness)
            position_history.append(best_position)

        best_index = np.argmin(fitness_values)
        best_solution = self._decode(population[best_index : best_index + 1])[0]
        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=fitness_values[best_index],
            fitness_history=fitness_history,
            position_history=position_history,
        )
