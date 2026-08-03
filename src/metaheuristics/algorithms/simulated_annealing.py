from collections.abc import Callable, Sequence

import numpy as np

from metaheuristics.result import OptimizationResult


class SimulatedAnnealing:
    """Continuous Simulated Annealing (Kirkpatrick, Gelatt & Vecchi, 1983).

    At each iteration a candidate is drawn from a Gaussian neighborhood of
    the current point, $x' = x + \\mathcal{N}(0, \\sigma^2)$, and accepted
    with the Metropolis criterion:

    $$P(\\text{accept}) = \\begin{cases} 1 & f(x') < f(x) \\\\ \\exp\\left(-\\frac{f(x') - f(x)}{T}\\right) & \\text{otherwise} \\end{cases}$$

    The temperature $T$ decays geometrically, $T \\leftarrow \\alpha T$,
    trading exploration (accepting worse moves) for exploitation as the run
    progresses.

    Parameters
    ----------
    max_iterations : int
        Number of proposal/accept steps to run.
    initial_temperature : float
        Starting temperature $T_0$.
    cooling_rate : float
        Geometric cooling factor $\\alpha \\in (0, 1)$ applied each iteration.
    step_fraction : float
        Neighborhood standard deviation, as a fraction of each dimension's
        bound span.
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        initial_temperature: float = 10.0,
        cooling_rate: float = 0.995,
        step_fraction: float = 0.1,
    ) -> None:
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.step_fraction = step_fraction

    def optimize(
        self, objective_fn: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]]
    ) -> OptimizationResult:
        """Run simulated annealing to minimize ``objective_fn`` over ``bounds``."""
        bounds_arr = np.asarray(bounds, dtype=float)
        low, high = bounds_arr[:, 0], bounds_arr[:, 1]
        step_sigma = self.step_fraction * (high - low)

        current = low + np.random.rand(len(bounds_arr)) * (high - low)
        current_fitness = objective_fn(current)

        best = current.copy()
        best_fitness = current_fitness
        temperature = self.initial_temperature
        fitness_history, position_history = [best_fitness], [best.copy()]

        for _ in range(self.max_iterations):
            candidate = np.clip(current + np.random.normal(0, step_sigma), low, high)
            candidate_fitness = objective_fn(candidate)

            delta = candidate_fitness - current_fitness
            if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
                current, current_fitness = candidate, candidate_fitness
                if current_fitness < best_fitness:
                    best, best_fitness = current.copy(), current_fitness

            temperature *= self.cooling_rate
            fitness_history.append(best_fitness)
            position_history.append(best.copy())

        return OptimizationResult(
            best_solution=best,
            best_fitness=best_fitness,
            fitness_history=fitness_history,
            position_history=position_history,
        )
