from collections.abc import Callable, Sequence

import numpy as np

from metaheuristics.result import OptimizationResult


class SlimeMouldAlgorithm:
    """Slime Mould Algorithm, SMA (Li, Chen, Deng & Mirjalili, 2020).

    Mimics the foraging behavior of *Physarum polycephalum*: each agent
    either approaches the best-known food source, weighted by how strongly
    it "smells" it, or oscillates/contracts around its own position:

    $$
    X(t+1) = \\begin{cases}
        X_b(t) + \\vec{vb} \\cdot (W \\cdot X_A(t) - X_B(t)) & r < p \\\\
        \\vec{vc} \\cdot X(t) & r \\geq p
    \\end{cases}
    $$

    $p = \\tanh(|S(i) - bF|)$ measures how close agent $i$'s fitness $S(i)$
    is to the best fitness found so far $bF$; $A, B$ are two other random
    agents; $W$ is a "weight" that grows for agents close to the food
    source and shrinks for those far from it; $\\vec{vb}$ and $\\vec{vc}$ are
    per-iteration random vectors whose ranges shrink over the run, focusing
    the search. With small probability $z$, an agent is re-randomized
    within the bounds instead, for exploration.

    Parameters
    ----------
    population_size : int
        Number of slime mould agents.
    max_iterations : int
        Number of iterations to run.
    exploration_rate : float
        Probability $z$ of randomly re-initializing an agent each iteration.
    """

    def __init__(
        self,
        population_size: int = 40,
        max_iterations: int = 100,
        exploration_rate: float = 0.03,
    ) -> None:
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.exploration_rate = exploration_rate

    def _weights(self, fitness: np.ndarray) -> np.ndarray:
        """Compute the smell-based weight of each agent (better half grows, worse half shrinks)."""
        order = np.argsort(fitness)
        best_fitness, worst_fitness = fitness[order[0]], fitness[order[-1]]
        span = (best_fitness - worst_fitness) or 1e-10

        weights = np.empty_like(fitness)
        r = np.random.rand(len(fitness))
        log_term = np.log10((best_fitness - fitness) / span + 1)
        half = len(fitness) // 2
        better_half = order[:half]
        worse_half = order[half:]
        weights[better_half] = 1 + r[better_half] * log_term[better_half]
        weights[worse_half] = 1 - r[worse_half] * log_term[worse_half]
        return weights

    def optimize(
        self, objective_fn: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]]
    ) -> OptimizationResult:
        """Run the Slime Mould Algorithm to minimize ``objective_fn`` over ``bounds``."""
        bounds_arr = np.asarray(bounds, dtype=float)
        low, high = bounds_arr[:, 0], bounds_arr[:, 1]
        num_dims = len(bounds_arr)

        population = low + np.random.rand(self.population_size, num_dims) * (high - low)
        fitness = np.array([objective_fn(x) for x in population])

        best_index = np.argmin(fitness)
        best_position = population[best_index].copy()
        best_fitness = fitness[best_index]
        fitness_history, position_history = [best_fitness], [best_position.copy()]
        population_history = [population.copy()]

        for iteration in range(self.max_iterations):
            progress = (iteration + 1) / self.max_iterations
            a = np.arctanh(np.clip(1 - progress, -0.999999, 0.999999))
            vc_bound = 1 - progress

            weights = self._weights(fitness)
            p = np.tanh(np.abs(fitness - fitness.min()))

            new_population = population.copy()
            for i in range(self.population_size):
                if np.random.rand() < self.exploration_rate:
                    new_population[i] = low + np.random.rand(num_dims) * (high - low)
                elif np.random.rand() < p[i]:
                    vb = np.random.uniform(-a, a, size=num_dims)
                    idx_a, idx_b = np.random.choice(self.population_size, 2, replace=False)
                    new_population[i] = best_position + vb * (
                        weights[i] * population[idx_a] - population[idx_b]
                    )
                else:
                    vc = np.random.uniform(-vc_bound, vc_bound, size=num_dims)
                    new_population[i] = vc * population[i]
                new_population[i] = np.clip(new_population[i], low, high)

            population = new_population
            fitness = np.array([objective_fn(x) for x in population])

            gen_best_index = np.argmin(fitness)
            if fitness[gen_best_index] < best_fitness:
                best_fitness = fitness[gen_best_index]
                best_position = population[gen_best_index].copy()
            fitness_history.append(best_fitness)
            position_history.append(best_position.copy())
            population_history.append(population.copy())

        return OptimizationResult(
            best_solution=best_position,
            best_fitness=best_fitness,
            fitness_history=fitness_history,
            position_history=position_history,
            population_history=population_history,
        )
