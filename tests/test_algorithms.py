import itertools

import numpy as np
import pytest

from metaheuristics.algorithms.differential_evolution import DifferentialEvolution
from metaheuristics.algorithms.genetic_algorithm import GeneticAlgorithm
from metaheuristics.algorithms.particle_swarm import ParticleSwarmOptimization
from metaheuristics.algorithms.simulated_annealing import SimulatedAnnealing
from metaheuristics.algorithms.slime_mould import SlimeMouldAlgorithm
from metaheuristics.benchmarks.landscapes import sphere

ALGORITHMS = {
    "differential_evolution": lambda: DifferentialEvolution(population_size=30, max_generations=60),
    "genetic_algorithm": lambda: GeneticAlgorithm(population_size=40, max_generations=60),
    "particle_swarm": lambda: ParticleSwarmOptimization(num_particles=30, max_iterations=60),
    "simulated_annealing": lambda: SimulatedAnnealing(max_iterations=2000),
    "slime_mould": lambda: SlimeMouldAlgorithm(population_size=30, max_iterations=60),
}


@pytest.mark.parametrize("name,make_algorithm", sorted(ALGORITHMS.items()))
def test_converges_on_sphere(name, make_algorithm):
    np.random.seed(42)
    result = make_algorithm().optimize(sphere, sphere.bounds)
    assert result.best_fitness < 1.0  # loose smoke threshold, not exact reproducibility
    assert len(result.fitness_history) == len(result.position_history)


@pytest.mark.parametrize("name,make_algorithm", sorted(ALGORITHMS.items()))
def test_fitness_history_is_best_so_far(name, make_algorithm):
    np.random.seed(0)
    result = make_algorithm().optimize(sphere, sphere.bounds)
    history = result.fitness_history
    assert all(a >= b - 1e-9 for a, b in itertools.pairwise(history))
