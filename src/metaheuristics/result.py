from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimizationResult:
    """Outcome of a metaheuristic run.

    Attributes
    ----------
    best_solution : np.ndarray
        The best decision vector found.
    best_fitness : float
        Objective value at ``best_solution``.
    fitness_history : list[float]
        Best-so-far fitness at each iteration, for convergence plots.
    position_history : list[np.ndarray]
        Best-so-far position at each iteration, for trajectory animations
        over a benchmark landscape.
    """

    best_solution: np.ndarray
    best_fitness: float
    fitness_history: list = field(default_factory=list)
    position_history: list = field(default_factory=list)
