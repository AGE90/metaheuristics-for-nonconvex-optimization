# Benchmark Functions Reference

Implementation: [`src/metaheuristics/benchmarks/landscapes.py`](../../src/metaheuristics/benchmarks/landscapes.py)
Notebook: [`notebooks/01_benchmark_landscapes.ipynb`](../../notebooks/01_benchmark_landscapes.ipynb)
Source: curated subset of [Wikipedia: Test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

All functions here are 2D, posed as **minimization** problems, and each carries `.bounds` (the
conventional search domain) and `.global_minimum` (`x*`, `f(x*)`) as attributes — used directly by
`tests/test_benchmarks.py` and the optimizer notebooks.

| Function | Category | Formula | Domain | Global minimum |
|---|---|---|---|---|
| `sphere` | bowl-shaped, unimodal | $\sum_i x_i^2$ | $[-5.12, 5.12]^2$ | $f(0,0)=0$ |
| `booth` | bowl-shaped, simple | $(x+2y-7)^2+(2x+y-5)^2$ | $[-10,10]^2$ | $f(1,3)=0$ |
| `matyas` | plate-shaped, smooth | $0.26(x^2+y^2)-0.48xy$ | $[-10,10]^2$ | $f(0,0)=0$ |
| `rosenbrock` | valley-shaped | $(1-x)^2+100(y-x^2)^2$ | $[-2,2]\times[-1,3]$ | $f(1,1)=0$ |
| `beale` | multimodal, narrow valley | $(1.5-x+xy)^2+(2.25-x+xy^2)^2+(2.625-x+xy^3)^2$ | $[-4.5,4.5]^2$ | $f(3,0.5)=0$ |
| `rastrigin` | many local minima | $10\cdot2+\sum_i[x_i^2-10\cos(2\pi x_i)]$ | $[-5.12,5.12]^2$ | $f(0,0)=0$ |
| `ackley` | many local minima | $-20e^{-0.2\sqrt{0.5(x^2+y^2)}}-e^{0.5(\cos2\pi x+\cos2\pi y)}+e+20$ | $[-5,5]^2$ | $f(0,0)=0$ |
| `griewank` | many local minima, non-separable | $\sum_i \tfrac{x_i^2}{4000}-\prod_i\cos\!\big(\tfrac{x_i}{\sqrt{i+1}}\big)+1$ | $[-600,600]^2$ | $f(0,0)=0$ |
| `levi13` | many local minima | $\sin^2(3\pi x)+(x-1)^2(1+\sin^2 3\pi y)+(y-1)^2(1+\sin^2 2\pi y)$ | $[-10,10]^2$ | $f(1,1)=0$ |
| `himmelblau` | multiple global minima | $(x^2+y-11)^2+(x+y^2-7)^2$ | $[-5,5]^2$ | $f(3,2)=0$ (4 minima) |
| `easom` | steep needle optimum | $-\cos x\cos y\exp(-((x-\pi)^2+(y-\pi)^2))$ | $[-100,100]^2$ | $f(\pi,\pi)=-1$ |
| `eggholder` | extremely rugged | $-(y+47)\sin\sqrt{\lvert x/2+(y+47)\rvert}-x\sin\sqrt{\lvert x-(y+47)\rvert}$ | $[-512,512]^2$ | $f(512, 404.23)\approx-959.64$ |

## Reading the categories

- **Bowl/plate-shaped**: smooth, mostly convex-looking — every algorithm here should solve these
  reliably; useful as a sanity check when debugging a new algorithm.
- **Valley-shaped**: a narrow curved region contains the optimum (Rosenbrock's "banana"); tests
  whether an algorithm can follow a valley rather than just descend the nearest slope.
- **Many local minima**: dense fields of local optima test escape from premature convergence.
- **Multiple global minima**: tests whether a run explores enough to notice equally-good
  alternatives, or collapses onto whichever basin it finds first.
- **Steep/rugged**: Easom (needle in a huge flat plateau) and Eggholder (chaotic, high-frequency
  ruggedness) are deliberately close to worst-case for local-search-flavored methods.
