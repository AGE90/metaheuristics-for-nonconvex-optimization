from collections.abc import Callable, Sequence

import numpy as np

from metaheuristics.result import OptimizationResult


class DifferentialEvolution:
    """Differential Evolution, DE/rand/1/bin (Storn & Price, 1997).

    For each target vector $x_i$, a mutant is built from three other
    distinct population members $a, b, c$:

    $$v = a + F (b - c)$$

    A trial vector is then formed by binomial crossover between $x_i$ and
    $v$ (each coordinate taken from $v$ with probability $CR$, with at
    least one coordinate guaranteed from $v$), and the trial replaces
    $x_i$ if it is at least as good.

    Parameters
    ----------
    population_size : int
        Number of vectors in the population.
    max_generations : int
        Number of generations to run.
    differential_weight : float
        Mutation scale factor $F \\in [0, 2]$.
    crossover_rate : float
        Binomial crossover probability $CR \\in [0, 1]$.
    """

    def __init__(
        self,
        population_size: int = 40,
        max_generations: int = 100,
        differential_weight: float = 0.8,
        crossover_rate: float = 0.9,
    ) -> None:
        self.population_size = population_size
        self.max_generations = max_generations
        self.differential_weight = differential_weight
        self.crossover_rate = crossover_rate

    def optimize(
        self, objective_fn: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]]
    ) -> OptimizationResult:
        """Run differential evolution to minimize ``objective_fn`` over ``bounds``."""
        bounds_arr = np.asarray(bounds, dtype=float)
        low, high = bounds_arr[:, 0], bounds_arr[:, 1]
        num_dims = len(bounds_arr)

        population = low + np.random.rand(self.population_size, num_dims) * (high - low)
        fitness = np.array([objective_fn(x) for x in population])

        best_index = np.argmin(fitness)
        fitness_history = [fitness[best_index]]
        position_history = [population[best_index].copy()]
        population_history = [population.copy()]

        for _ in range(self.max_generations):
            for i in range(self.population_size):
                candidates = np.random.choice(
                    [j for j in range(self.population_size) if j != i], 3, replace=False
                )
                a, b, c = population[candidates]
                mutant = np.clip(a + self.differential_weight * (b - c), low, high)

                cross_mask = np.random.rand(num_dims) < self.crossover_rate
                cross_mask[np.random.randint(num_dims)] = True
                trial = np.where(cross_mask, mutant, population[i])

                trial_fitness = objective_fn(trial)
                if trial_fitness <= fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness

            best_index = np.argmin(fitness)
            fitness_history.append(fitness[best_index])
            position_history.append(population[best_index].copy())
            population_history.append(population.copy())

        best_index = np.argmin(fitness)
        return OptimizationResult(
            best_solution=population[best_index],
            best_fitness=fitness[best_index],
            fitness_history=fitness_history,
            position_history=position_history,
            population_history=population_history,
        )
