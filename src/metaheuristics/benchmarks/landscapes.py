"""2D benchmark (test) functions for nonconvex optimization.

Each function takes ``x``, either a length-2 array ``[x1, x2]`` (single
point) or a stacked array of shape ``(2, ...)`` (e.g. ``np.array([X1, X2])``
from ``np.meshgrid``, for vectorized surface plotting), and returns the
scalar objective value(s). All functions are posed as **minimization**
problems, matching the optimizer modules in ``metaheuristics.algorithms``.

Each function is wrapped in a :class:`BenchmarkFunction`, a small callable dataclass carrying two
attributes used by tests, docs, and notebooks:

- ``bounds``: the conventional search domain, as ``((x1_lo, x1_hi), (x2_lo, x2_hi))``.
- ``global_minimum``: ``(x_star, f_star)``, a known minimizer and its value.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class BenchmarkFunction:
    """Callable wrapper pairing a landscape function with its bounds and known optimum."""

    func: Callable[[np.ndarray], np.ndarray]
    bounds: tuple[tuple[float, float], tuple[float, float]]
    global_minimum: tuple[np.ndarray, float]

    def __call__(self, x):
        return self.func(x)

    @property
    def __name__(self) -> str:
        return self.func.__name__


def sphere(x):
    """$f(\\mathbf{x}) = \\sum_i x_i^2$"""
    x = np.asarray(x)
    return np.sum(x**2, axis=0)


sphere = BenchmarkFunction(sphere, ((-5.12, 5.12), (-5.12, 5.12)), (np.zeros(2), 0.0))


def rastrigin(x):
    """$f(\\mathbf{x}) = An + \\sum_i \\left[x_i^2 - A\\cos(2\\pi x_i)\\right]$, $A=10$"""
    x1, x2 = np.asarray(x)
    n = 2
    A = x1**2 - 10 * np.cos(2 * np.pi * x1)
    B = x2**2 - 10 * np.cos(2 * np.pi * x2)
    return 10 * n + A + B


rastrigin = BenchmarkFunction(rastrigin, ((-5.12, 5.12), (-5.12, 5.12)), (np.zeros(2), 0.0))


def ackley(x):
    """$f(x,y) = -20e^{-0.2\\sqrt{0.5(x^2+y^2)}} - e^{0.5(\\cos 2\\pi x + \\cos 2\\pi y)} + e + 20$"""
    x1, x2 = np.asarray(x)
    A = np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    B = np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    return -20.0 * A - B + np.exp(1) + 20


ackley = BenchmarkFunction(ackley, ((-5.0, 5.0), (-5.0, 5.0)), (np.zeros(2), 0.0))


def rosenbrock(x):
    """$f(x,y) = (a-x)^2 + b(y-x^2)^2$, $a=1, b=100$"""
    x1, x2 = np.asarray(x)
    a, b = 1, 100
    return (a - x1) ** 2 + b * (x2 - x1**2) ** 2


rosenbrock = BenchmarkFunction(rosenbrock, ((-2.0, 2.0), (-1.0, 3.0)), (np.array([1.0, 1.0]), 0.0))


def beale(x):
    """$f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2$"""
    x1, x2 = np.asarray(x)
    A = (1.5 - x1 + x1 * x2) ** 2
    B = (2.25 - x1 + x1 * x2**2) ** 2
    C = (2.625 - x1 + x1 * x2**3) ** 2
    return A + B + C


beale = BenchmarkFunction(beale, ((-4.5, 4.5), (-4.5, 4.5)), (np.array([3.0, 0.5]), 0.0))


def griewank(x):
    """$f(\\mathbf{x}) = \\sum_i \\frac{x_i^2}{4000} - \\prod_i \\cos\\left(\\frac{x_i}{\\sqrt{i+1}}\\right) + 1$"""
    x = np.asarray(x)
    indices = np.arange(1, x.shape[0] + 1).reshape((-1,) + (1,) * (x.ndim - 1))
    return np.sum(x**2, axis=0) / 4000 - np.prod(np.cos(x / np.sqrt(indices)), axis=0) + 1


griewank = BenchmarkFunction(griewank, ((-600.0, 600.0), (-600.0, 600.0)), (np.zeros(2), 0.0))


def himmelblau(x):
    """$f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2$ -- four global minima"""
    x1, x2 = np.asarray(x)
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


himmelblau = BenchmarkFunction(himmelblau, ((-5.0, 5.0), (-5.0, 5.0)), (np.array([3.0, 2.0]), 0.0))


def easom(x):
    """$f(x,y) = -\\cos(x)\\cos(y)\\exp\\left(-((x-\\pi)^2+(y-\\pi)^2)\\right)$ -- steep needle optimum"""
    x1, x2 = np.asarray(x)
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2))


easom = BenchmarkFunction(easom, ((-100.0, 100.0), (-100.0, 100.0)), (np.array([np.pi, np.pi]), -1.0))


def eggholder(x):
    """$f(x,y) = -(y+47)\\sin\\sqrt{|x/2+(y+47)|} - x\\sin\\sqrt{|x-(y+47)|}$ -- extremely rugged"""
    x1, x2 = np.asarray(x)
    A = -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47))))
    B = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    return A + B


eggholder = BenchmarkFunction(
    eggholder, ((-512.0, 512.0), (-512.0, 512.0)), (np.array([512.0, 404.2319]), -959.6407)
)


def booth(x):
    """$f(x,y) = (x+2y-7)^2 + (2x+y-5)^2$"""
    x1, x2 = np.asarray(x)
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


booth = BenchmarkFunction(booth, ((-10.0, 10.0), (-10.0, 10.0)), (np.array([1.0, 3.0]), 0.0))


def matyas(x):
    """$f(x,y) = 0.26(x^2+y^2) - 0.48xy$ -- plate-shaped"""
    x1, x2 = np.asarray(x)
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


matyas = BenchmarkFunction(matyas, ((-10.0, 10.0), (-10.0, 10.0)), (np.zeros(2), 0.0))


def levi13(x):
    """$f(x,y) = \\sin^2(3\\pi x) + (x-1)^2(1+\\sin^2(3\\pi y)) + (y-1)^2(1+\\sin^2(2\\pi y))$"""
    x1, x2 = np.asarray(x)
    A = np.sin(3 * np.pi * x1) ** 2
    B = (x1 - 1) ** 2 * (1 + np.sin(3 * np.pi * x2) ** 2)
    C = (x2 - 1) ** 2 * (1 + np.sin(2 * np.pi * x2) ** 2)
    return A + B + C


levi13 = BenchmarkFunction(levi13, ((-10.0, 10.0), (-10.0, 10.0)), (np.array([1.0, 1.0]), 0.0))


ALL_FUNCTIONS = {
    f.__name__: f
    for f in (
        sphere,
        rastrigin,
        ackley,
        rosenbrock,
        beale,
        griewank,
        himmelblau,
        easom,
        eggholder,
        booth,
        matyas,
        levi13,
    )
}
