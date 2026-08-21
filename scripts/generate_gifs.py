"""Generate the README's algorithm/benchmark GIF gallery into assets/.

Requires the ``viz`` dependency group: ``uv run --group viz python scripts/generate_gifs.py``.
"""

from pathlib import Path

import numpy as np

from metaheuristics.algorithms.differential_evolution import DifferentialEvolution
from metaheuristics.algorithms.genetic_algorithm import GeneticAlgorithm
from metaheuristics.algorithms.particle_swarm import ParticleSwarmOptimization
from metaheuristics.algorithms.simulated_annealing import SimulatedAnnealing
from metaheuristics.algorithms.slime_mould import SlimeMouldAlgorithm
from metaheuristics.benchmarks import landscapes
from metaheuristics.viz.plotly_surfaces import render_search_animation_gif

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

COMBOS = [
    ("pso_ackley.gif", ParticleSwarmOptimization(num_particles=30, max_iterations=60), landscapes.ackley),
    ("ga_rastrigin.gif", GeneticAlgorithm(population_size=40, max_generations=60), landscapes.rastrigin),
    ("sa_rastrigin.gif", SimulatedAnnealing(max_iterations=400), landscapes.rastrigin),
    ("de_rosenbrock.gif", DifferentialEvolution(population_size=30, max_generations=60), landscapes.rosenbrock),
    ("sma_himmelblau.gif", SlimeMouldAlgorithm(population_size=30, max_iterations=60), landscapes.himmelblau),
]


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    for filename, optimizer, func in COMBOS:
        out_path = ASSETS_DIR / filename
        if out_path.exists():
            print(f"Skipping {out_path}, already exists")
            continue
        np.random.seed(0)
        result = optimizer.optimize(func, func.bounds)
        print(f"Rendering {out_path} ({type(optimizer).__name__} on {func.__name__})...")
        render_search_animation_gif(
            func,
            result,
            out_path=str(out_path),
            title=f"{type(optimizer).__name__} on {func.__name__}",
        )


if __name__ == "__main__":
    main()
