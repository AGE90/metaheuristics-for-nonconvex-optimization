# Particle Swarm Optimization (PSO)

Implementation: [`src/metaheuristics/algorithms/particle_swarm.py`](../../src/metaheuristics/algorithms/particle_swarm.py)
Notebook: [`notebooks/03_particle_swarm_optimization.ipynb`](../../notebooks/03_particle_swarm_optimization.ipynb)
Reference: Kennedy & Eberhart, *Particle Swarm Optimization* (1995); Shi & Eberhart, *A Modified
Particle Swarm Optimizer* (1998).

## Idea

Model the population as a swarm of particles moving through the search space with a position and
a velocity. Each particle is pulled towards two attractors: the best position *it personally* has
found ($p_i$), and the best position *any* particle in the swarm has found ($g$). No
selection/crossover/mutation — the whole population moves together, continuously.

## Update rule

$$v_i \leftarrow w\, v_i + c_1 r_1 (p_i - x_i) + c_2 r_2 (g - x_i)$$
$$x_i \leftarrow x_i + v_i$$

where $r_1, r_2 \sim U(0,1)$ (independent per dimension) inject stochasticity, $c_1$
(`cognitive_coeff`) controls how strongly a particle trusts its own memory, and $c_2$
(`social_coeff`) how strongly it trusts the swarm. Positions are clipped back into `bounds` after
each step.

## Inertia weight

The inertia weight $w$ decays linearly from `inertia_start` to `inertia_end` over the run (Shi &
Eberhart, 1998):

$$w(t) = w_{\text{start}} + (w_{\text{end}} - w_{\text{start}}) \cdot \frac{t}{T-1}$$

High inertia early on preserves momentum (exploration); low inertia late in the run damps
oscillation so the swarm can settle (exploitation).

## When it does well / poorly

Very effective on smooth, unimodal-ish or moderately multimodal landscapes (e.g. Ackley) — the
whole swarm can converge quickly once a promising basin is found. Prone to premature convergence
on landscapes with many well-separated, deep local minima (Eggholder is the extreme case here):
once the swarm's global best locks onto a basin, every particle is pulled toward it, and momentum
does not carry particles far enough to discover better basins the same way binary
mutation/crossover (GA) or random re-initialization (SMO's exploration term) can.
