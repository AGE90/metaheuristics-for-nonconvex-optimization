# Genetic Algorithm (GA)

Implementation: [`src/metaheuristics/algorithms/genetic_algorithm.py`](../../src/metaheuristics/algorithms/genetic_algorithm.py)
Notebook: [`notebooks/02_genetic_algorithm.ipynb`](../../notebooks/02_genetic_algorithm.ipynb)
Reference: Holland, *Adaptation in Natural and Artificial Systems* (1975).

## Idea

Maintain a population of candidate solutions encoded as chromosomes, and evolve it generation by
generation using operators loosely modeled on natural selection: **selection** (better solutions
are more likely to reproduce), **crossover** (offspring combine material from two parents), and
**mutation** (small random perturbations maintain diversity).

## Real-valued encoding

Benchmark functions here take real vectors, so each variable $x_i \in [lb_i, ub_i]$ is encoded as
an $L$-bit unsigned integer and decoded linearly:

$$x_i = lb_i + \frac{\text{int}(b_i)}{2^L - 1}(ub_i - lb_i)$$

Bit count $L$ (`bits_per_variable`) trades off resolution against search-space size: with $L=16$,
each variable can take $2^{16}$ distinct values.

## Per-generation loop

1. **Tournament selection**: sample `tournament_size` individuals uniformly at random, keep the
   one with the lowest (best) fitness. Repeat once per parent slot.
2. **Single-point crossover**: for each consecutive parent pair, with probability
   `crossover_rate`, swap the bit-tails of both chromosomes after a random cut point.
3. **Bit-flip mutation**: flip each bit independently with probability `mutation_rate`.
4. **Elitism** (optional): the `elite_size` best individuals from the *previous* generation
   directly replace the worst individuals in the new offspring, guaranteeing fitness never
   regresses generation-over-generation.

## Hyperparameters that matter most

- `population_size`: bigger populations explore more per generation but cost more evaluations.
- `mutation_rate`: too low and the population converges prematurely to a local optimum; too high
  and it never converges (effectively random search). A common rule of thumb is $\approx 1/L$.
- `elite_size`: without elitism, a good solution can be lost to unlucky
  crossover/mutation — but too much elitism reduces diversity.

## When it does well / poorly

Does well on landscapes with multiple, well-separated basins, since crossover can combine
building blocks from different regions. Struggles on landscapes needing fine local
exploitation once near an optimum (e.g. narrow valleys like Rosenbrock) purely because of the
resolution limit of a fixed-bit encoding — Differential Evolution (see
[differential-evolution.md](differential-evolution.md)) tends to do better there.
