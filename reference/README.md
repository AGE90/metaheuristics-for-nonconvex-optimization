# Reference material

`papers/` is a place to drop PDFs of source papers as you collect them (gitignored contents aside
from `.gitkeep` — add your own copies locally). The citations below are the foundational papers
behind each algorithm in `src/metaheuristics/algorithms/`, plus other well-known Python
metaheuristics libraries found while researching this project, for when the from-scratch
implementations here aren't enough.

## Foundational papers

- **Genetic Algorithm** — Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*.
  University of Michigan Press.
- **Particle Swarm Optimization** — Kennedy, J., & Eberhart, R. (1995). *Particle Swarm
  Optimization*. Proceedings of ICNN'95. Inertia weight refinement: Shi, Y., & Eberhart, R.
  (1998). *A Modified Particle Swarm Optimizer*.
- **Simulated Annealing** — Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). *Optimization
  by Simulated Annealing*. Science, 220(4598).
- **Differential Evolution** — Storn, R., & Price, K. (1997). *Differential Evolution — A Simple
  and Efficient Heuristic for Global Optimization over Continuous Spaces*. Journal of Global
  Optimization, 11(4).
- **Slime Mould Algorithm** — Li, S., Chen, H., Wang, M., Heidari, A. A., & Mirjalili, S. (2020).
  *Slime Mould Algorithm: A New Method for Stochastic Optimization*. Future Generation Computer
  Systems, 111.

## Other Python metaheuristics libraries (not dependencies of this project)

Used as comparison baselines in `notebooks/07_library_comparison.ipynb`:

- [mealpy](https://mealpy.readthedocs.io/) — large collection of nature-inspired algorithms
  (GA, PSO, SA, DE, SMA and 150+ more) under one consistent API.
- [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) — `differential_evolution`
  and `dual_annealing` global optimizers, ubiquitous and already a dependency here for `scipy.signal`.

Not used here, but worth knowing about if this project's scope grows:

- [pyswarms](https://pyswarms.readthedocs.io/) — PSO-specific research toolkit.
- [DEAP](https://deap.readthedocs.io/) — flexible evolutionary computation framework, widely used
  as a GA baseline in the literature.
- [pymoo](https://pymoo.org/) — the standard choice once you need actual multi-objective/Pareto
  optimization rather than single-objective.
- [opfunu](https://opfunu.readthedocs.io/) — 300+ benchmark functions including all CEC
  competition suites, if the curated 12 in `benchmarks/landscapes.py` stop being enough.
- [PyPop7](https://pypop.readthedocs.io/) — pure-Python, aimed at large-scale black-box
  optimization.
