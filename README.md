# Metaheuristics for Nonconvex Optimization

A study companion and experimentation playground for metaheuristic optimization algorithms
applied to nonconvex, multimodal functions — built for learning how Genetic Algorithms, Particle
Swarm Optimization, Simulated Annealing, Differential Evolution, and the Slime Mould Algorithm
behave, why they differ, and where each one shines or struggles.

## What's here

- **`src/metaheuristics/`** — an installable package with from-scratch implementations of five
  algorithms, twelve 2D benchmark/test functions, and interactive Plotly visualization helpers.
- **`notebooks/`** — one notebook per algorithm (surface + convergence + trajectory animation),
  a from-scratch-vs-library comparison notebook, and `notebooks/applications/` with signal
  processing and data science examples.
- **`docs/notes/`** — short study notes per algorithm (the update rule, key hyperparameters, when
  it does well/poorly) and a benchmark-functions reference table.
- **`reference/`** — citations for the foundational papers and a place to keep source PDFs.
- **`tests/`** — a small pytest suite: benchmark functions evaluate correctly at their known
  optima, and each algorithm converges on the Sphere function within a loose smoke-test budget.

## Algorithms

| Algorithm | Module | Notes |
|---|---|---|
| Genetic Algorithm | `algorithms/genetic_algorithm.py` | Binary-encoded, decodes to real vectors |
| Particle Swarm Optimization | `algorithms/particle_swarm.py` | Inertia-weight variant (Shi & Eberhart, 1998) |
| Simulated Annealing | `algorithms/simulated_annealing.py` | Single-point search, not population-based |
| Differential Evolution | `algorithms/differential_evolution.py` | DE/rand/1/bin |
| Slime Mould Algorithm | `algorithms/slime_mould.py` | Li et al. (2020) |

All five share one interface, so they're interchangeable in notebooks and comparisons:

```python
from metaheuristics.algorithms.particle_swarm import ParticleSwarmOptimization
from metaheuristics.benchmarks.landscapes import rastrigin

result = ParticleSwarmOptimization(num_particles=40, max_iterations=100).optimize(
    rastrigin, rastrigin.bounds
)
result.best_solution, result.best_fitness  # np.ndarray, float
result.fitness_history, result.position_history  # per-iteration, for convergence/trajectory plots
```

`objective_fn` is a plain `f(x) -> float` over an $n$-dimensional real vector — the benchmark
functions in `metaheuristics.benchmarks.landscapes` are 2D, but the algorithms themselves are not
limited to 2D (see `notebooks/applications/fir_filter_design.ipynb`, a 21-dimensional problem).

Every function in `metaheuristics.benchmarks.landscapes` carries known metadata:

```python
rastrigin.bounds            # ((-5.12, 5.12), (-5.12, 5.12))
rastrigin.global_minimum    # (array([0., 0.]), 0.0)
```

See `docs/notes/benchmark-functions.md` for the full table (formula, domain, global minimum) of
all twelve functions, spanning bowl-shaped, valley-shaped, many-local-minima, multiple-global-
minima, and steep/rugged landscapes.

## Quickstart

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
uv sync                 # install the package + all dependencies into .venv
uv run pytest           # run the test suite
uv run jupyter lab      # open the notebooks
```

`uv sync` installs `metaheuristics` itself in editable mode, so notebooks and tests can
`import metaheuristics...` directly — no `sys.path` hacks.

## Math notation

Docs and notebooks use LaTeX (`$...$` inline, `$$...$$` display) for update rules and formulas,
e.g. PSO's velocity update:

$$v_i \leftarrow w\, v_i + c_1 r_1 (p_i - x_i) + c_2 r_2 (g - x_i)$$

renders in both Markdown files (GitHub/most viewers) and Jupyter markdown cells.
