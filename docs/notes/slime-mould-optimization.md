# Slime Mould Algorithm (SMA / SMO)

Implementation: [`src/metaheuristics/algorithms/slime_mould.py`](../../src/metaheuristics/algorithms/slime_mould.py)
Notebook: [`notebooks/06_slime_mould_optimization.ipynb`](../../notebooks/06_slime_mould_optimization.ipynb)
Reference: Li, Chen, Deng & Mirjalili, *Slime Mould Algorithm: A New Method for Stochastic
Optimization* (2020).

## Idea

Modeled on the foraging behavior of the slime mould *Physarum polycephalum*, which extends
oscillating veins toward food sources, reinforcing veins that find better food (analogous to
pheromone reinforcement in ant colony optimization) while weaker veins contract. Each agent
either moves toward the best-known food source — more strongly if it currently "smells" it
closely — or contracts/oscillates in place, with occasional fully random moves for exploration.

## Update rule

$$
X(t+1) = \begin{cases}
    X_b(t) + \vec{vb} \cdot (W \cdot X_A(t) - X_B(t)) & r < p \\
    \vec{vc} \cdot X(t) & r \geq p
\end{cases}
$$

- $X_b$ is the best position found so far; $A, B$ are two other randomly chosen agents.
- $p = \tanh(|S(i) - bF|)$: how close agent $i$'s fitness $S(i)$ is to the best fitness $bF$ found
  so far — agents near the optimum are more likely to take the "approach food" branch.
- $W$ (the smell/weight term) grows for the fitter half of the population and shrinks for the
  weaker half, so better agents pull harder on the search.
- $\vec{vb} \in [-a, a]$ and $\vec{vc} \in [-vc_{\max}, vc_{\max}]$ are per-iteration random
  vectors whose ranges shrink over the run ($a = \operatorname{artanh}(1 - t/T)$,
  $vc_{\max} = 1 - t/T$), narrowing the search as the run progresses — conceptually similar to
  PSO's decaying inertia weight.
- With small probability `exploration_rate` ($z$, e.g. 0.03), an agent ignores all of this and
  re-randomizes within the bounds — a direct exploration/diversity mechanism GA gets from
  mutation and PSO doesn't have an equivalent of.

## When it does well / poorly

The random-reinitialization term makes SMO relatively good at not collapsing onto a single basin
prematurely — useful on landscapes with multiple equally-good optima like Himmelblau, where a run
should be able to land on any of the four minima rather than always the same one (see
`notebooks/06_slime_mould_optimization.ipynb`). Like PSO, its update rule assumes a reasonably
well-behaved fitness landscape (the $p$/$W$ terms are smooth in $S(i)$) and can struggle when
fitness values shift by orders of magnitude across the population.
