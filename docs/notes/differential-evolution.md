# Differential Evolution (DE)

Implementation: [`src/metaheuristics/algorithms/differential_evolution.py`](../../src/metaheuristics/algorithms/differential_evolution.py)
Notebook: [`notebooks/05_differential_evolution.ipynb`](../../notebooks/05_differential_evolution.ipynb)
Reference: Storn & Price, *Differential Evolution — A Simple and Efficient Heuristic for Global
Optimization over Continuous Spaces* (1997).

## Idea

Like GA, DE evolves a population by generation, but its mutation operator is specific to
continuous spaces: instead of random perturbation, it uses the **vector difference between two
other population members** as the mutation step. Early on (population spread out), steps are
large; as the population converges, differences shrink and steps naturally get smaller — DE
self-adapts its step size without any explicit cooling/decay schedule.

## Update rule — DE/rand/1/bin

For each target vector $x_i$, pick three other distinct population members $a, b, c$ and form a
mutant:

$$v = a + F (b - c)$$

where $F \in [0, 2]$ (`differential_weight`) scales the difference. A trial vector is then formed
by **binomial crossover** between the target and the mutant — each coordinate is taken from $v$
with probability $CR$ (`crossover_rate`), with at least one coordinate guaranteed from $v$ so the
trial always differs from the target. The trial replaces $x_i$ if it is at least as good
(greedy, per-individual selection — unlike GA, there's no population-wide elitism step needed).

## Hyperparameters that matter most

- `differential_weight` ($F$): controls exploration/exploitation balance; $F \approx 0.5$–$0.9$ is
  typical.
- `crossover_rate` ($CR$): high $CR$ (close to 1) makes trials look more like the mutant (more
  exploratory); low $CR$ preserves more of the target (more exploitative, better for separable
  problems).

## When it does well / poorly

Strong on landscapes with curved, non-axis-aligned valleys (Rosenbrock is the textbook case) —
the difference-vector mutation naturally samples along whatever direction the population has
already spread out in, which tends to align with the valley. See its trajectory on Rosenbrock in
`notebooks/05_differential_evolution.ipynb`.
