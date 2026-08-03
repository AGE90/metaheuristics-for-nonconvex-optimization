# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A study companion and experimentation playground for metaheuristic optimization algorithms (GA,
PSO, SA, DE, Slime Mould) applied to nonconvex/multimodal test functions. `src/metaheuristics/` is
an installable package; `notebooks/` explore and visualize it; `docs/notes/` hold math-notated
study notes; `reference/` holds paper citations.

## Commands

```bash
uv sync                              # install package + all deps into .venv
uv run pytest                        # run the test suite
uv run pytest tests/test_algorithms.py -k pso   # run a single test file/case
uv run jupyter lab                   # open the notebooks
uv run ruff check .                  # lint
```

`uv sync` installs `metaheuristics` itself in editable mode, so notebooks/tests `import
metaheuristics...` directly — there's no `sys.path` manipulation anywhere in this repo.

## Architecture

- **`src/metaheuristics/algorithms/`** — five optimizer classes (`GeneticAlgorithm`,
  `ParticleSwarmOptimization`, `SimulatedAnnealing`, `DifferentialEvolution`,
  `SlimeMouldAlgorithm`), all sharing one interface:
  `optimizer.optimize(objective_fn, bounds) -> OptimizationResult`, where `objective_fn` is a
  plain `f(x: np.ndarray) -> float` (minimization) and `bounds` is a sequence of `(low, high)`
  pairs, one per dimension. This shared shape is what lets `notebooks/07_library_comparison.ipynb`
  and the viz helpers treat all five algorithms uniformly — keep it consistent if you add a sixth.
  `GeneticAlgorithm` is the odd one out internally: it keeps a binary genome and decodes to real
  vectors via `bounds` before evaluating `objective_fn` (see `_decode` in
  `genetic_algorithm.py`), so it can run on the same continuous benchmark functions as the other
  four despite being bit-string-based.
- **`src/metaheuristics/result.py`** — `OptimizationResult` (`best_solution`, `best_fitness`,
  `fitness_history`, `position_history`). `position_history` is the best-so-far point at each
  iteration, not just the fitness — it exists specifically to drive the trajectory animations in
  `viz/plotly_surfaces.py`, so don't drop it when touching an algorithm's return path.
- **`src/metaheuristics/benchmarks/landscapes.py`** — twelve 2D benchmark functions
  (`ALL_FUNCTIONS` dict). Each function is a plain callable with `.bounds` and `.global_minimum`
  attached as attributes (not a wrapper class) — `tests/test_benchmarks.py` and
  `docs/notes/benchmark-functions.md` both key off these. Functions accept either a length-2
  vector (single point, used by the optimizers) or a stacked `(2, ...)` array from `np.meshgrid`
  (vectorized, used for surface plotting) — see the module docstring for the exact convention.
- **`src/metaheuristics/viz/plotly_surfaces.py`** — `plot_surface` (3D landscape) and
  `plot_contour_with_trajectory` (animates `OptimizationResult.position_history` over a contour).
  The latter subsamples long histories to `max_frames` before building animation frames, since
  each frame's trail is cumulative — don't remove that subsampling, it's what keeps
  many-iteration runs (e.g. Simulated Annealing) from producing multi-hundred-MB notebooks.
- **`notebooks/`** — `01_benchmark_landscapes.ipynb` through `06_slime_mould_optimization.ipynb`
  are one-per-topic; `07_library_comparison.ipynb` runs the from-scratch implementations against
  `mealpy` (covers all five algorithms under one API) and `scipy.optimize` as baselines — see that
  notebook for the exact mealpy 3.x `Problem`/`FloatVar`/`model.solve(...)` API shape if adding
  another library comparison. `notebooks/applications/` are domain examples (FIR filter design,
  ML hyperparameter tuning, feature selection) that reuse the same optimizer classes on
  higher-dimensional, non-benchmark objectives.

## Working in this repo

- Notebooks are committed with their outputs (including rendered Plotly figures) so they're
  readable without re-running. After editing a notebook's code, re-execute it headlessly rather
  than hand-editing the JSON: `uv run jupyter nbconvert --to notebook --execute --inplace <path>`.
- Every algorithm and benchmark function has a corresponding test in `tests/` — keep it that way.
  `tests/test_algorithms.py` uses a loose smoke threshold on the Sphere function (seeded), not
  exact-reproducibility assertions; `tests/test_benchmarks.py` checks each function's known
  optimum. Both are meant to catch broken algorithm logic, not to benchmark quality.
