from collections.abc import Callable, Sequence

import numpy as np

from metaheuristics.result import OptimizationResult


class ParticleSwarmOptimization:
    """Particle Swarm Optimization (Kennedy & Eberhart, 1995) with inertia weight.

    Velocity update: $v_i \\leftarrow w v_i + c_1 r_1 (p_i - x_i) + c_2 r_2 (g - x_i)$
    Position update: $x_i \\leftarrow x_i + v_i$

    where $p_i$ is particle $i$'s personal best and $g$ the swarm's global
    best. The inertia weight $w$ decays linearly from ``inertia_start`` to
    ``inertia_end`` over the run (Shi & Eberhart, 1998), balancing
    exploration early on against exploitation later.

    Parameters
    ----------
    num_particles : int
        Swarm size.
    max_iterations : int
        Number of iterations to run.
    cognitive_coeff : float
        Pull towards each particle's own best position ($c_1$).
    social_coeff : float
        Pull towards the swarm's global best position ($c_2$).
    inertia_start, inertia_end : float
        Inertia weight at the first and last iteration.
    """

    def __init__(
        self,
        num_particles: int = 40,
        max_iterations: int = 100,
        cognitive_coeff: float = 1.5,
        social_coeff: float = 1.5,
        inertia_start: float = 0.9,
        inertia_end: float = 0.4,
    ) -> None:
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.inertia_start = inertia_start
        self.inertia_end = inertia_end

    def optimize(
        self, objective_fn: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]]
    ) -> OptimizationResult:
        """Run PSO to minimize ``objective_fn`` over ``bounds``."""
        bounds_arr = np.asarray(bounds, dtype=float)
        low, high = bounds_arr[:, 0], bounds_arr[:, 1]
        span = high - low
        num_dims = len(bounds_arr)

        positions = low + np.random.rand(self.num_particles, num_dims) * span
        velocities = (np.random.rand(self.num_particles, num_dims) - 0.5) * span

        personal_best_positions = positions.copy()
        personal_best_fitness = np.array([objective_fn(x) for x in positions])

        global_best_index = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_index].copy()
        global_best_fitness = personal_best_fitness[global_best_index]

        fitness_history, position_history = [global_best_fitness], [global_best_position.copy()]

        for iteration in range(self.max_iterations):
            inertia = self.inertia_start + (self.inertia_end - self.inertia_start) * (
                iteration / max(self.max_iterations - 1, 1)
            )
            r1 = np.random.rand(self.num_particles, num_dims)
            r2 = np.random.rand(self.num_particles, num_dims)

            velocities = (
                inertia * velocities
                + self.cognitive_coeff * r1 * (personal_best_positions - positions)
                + self.social_coeff * r2 * (global_best_position - positions)
            )
            positions = np.clip(positions + velocities, low, high)

            fitness = np.array([objective_fn(x) for x in positions])

            improved = fitness < personal_best_fitness
            personal_best_positions[improved] = positions[improved]
            personal_best_fitness[improved] = fitness[improved]

            best_index = np.argmin(personal_best_fitness)
            if personal_best_fitness[best_index] < global_best_fitness:
                global_best_position = personal_best_positions[best_index].copy()
                global_best_fitness = personal_best_fitness[best_index]

            fitness_history.append(global_best_fitness)
            position_history.append(global_best_position.copy())

        return OptimizationResult(
            best_solution=global_best_position,
            best_fitness=global_best_fitness,
            fitness_history=fitness_history,
            position_history=position_history,
        )
