# Simulated Annealing (SA)

Implementation: [`src/metaheuristics/algorithms/simulated_annealing.py`](../../src/metaheuristics/algorithms/simulated_annealing.py)
Notebook: [`notebooks/04_simulated_annealing.ipynb`](../../notebooks/04_simulated_annealing.ipynb)
Reference: Kirkpatrick, Gelatt & Vecchi, *Optimization by Simulated Annealing* (1983).

## Idea

Named for annealing in metallurgy: a metal cooled slowly settles into a low-energy crystal
structure, while a metal cooled too fast freezes into a higher-energy, defect-ridden state. SA
applies the same idea to optimization — a single point wanders the landscape, occasionally
accepting *worse* moves (to escape local minima), with that willingness decreasing ("cooling")
over the run.

**This is the only algorithm in this project that is not population-based** — a single trajectory,
not a population, does the searching. That makes it cheap per iteration but also the most likely
to get stuck or wander unproductively on a rugged, high-dimensional landscape.

## Update rule

At each step, propose a candidate from a Gaussian neighborhood of the current point:

$$x' = x + \mathcal{N}(0, \sigma^2), \qquad \sigma = \texttt{step\_fraction} \times (ub - lb)$$

and accept it with the Metropolis criterion:

$$P(\text{accept}) = \begin{cases} 1 & f(x') < f(x) \\ \exp\left(-\dfrac{f(x') - f(x)}{T}\right) & \text{otherwise} \end{cases}$$

The temperature cools geometrically each iteration, $T \leftarrow \alpha T$ (`cooling_rate`
$=\alpha$). High $T$ means almost any move is accepted (exploration); as $T \to 0$, only
improving moves are accepted (pure local search / hill descending).

## Hyperparameters that matter most

- `initial_temperature`: too low and the search behaves like greedy local descent from the start,
  never escaping the first basin it falls into.
- `cooling_rate`: too fast and the run "freezes" before finding a good region; too slow and it
  wastes iterations still accepting bad moves near the end of the budget.
- `step_fraction`: the neighborhood size — too large and most proposals land in clearly worse
  regions and get rejected; too small and each step barely moves.

## When it does well / poorly

Reasonable on smooth or mildly multimodal landscapes with a generous iteration budget (its
per-iteration cost is a single objective evaluation, so it can afford many more iterations than a
population method for the same budget). On strongly multimodal landscapes like Rastrigin or
Eggholder it is noticeably less reliable than the population-based algorithms in this project,
since there's no "swarm memory" — compare its convergence curve on Rastrigin in
`notebooks/04_simulated_annealing.ipynb` to GA's on the same function in
`notebooks/02_genetic_algorithm.ipynb`.
