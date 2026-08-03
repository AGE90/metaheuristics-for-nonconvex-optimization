import pytest

from metaheuristics.benchmarks.landscapes import ALL_FUNCTIONS


@pytest.mark.parametrize("name,func", sorted(ALL_FUNCTIONS.items()))
def test_global_minimum(name, func):
    x_star, f_star = func.global_minimum
    assert func(x_star) == pytest.approx(f_star, abs=1e-3)


@pytest.mark.parametrize("name,func", sorted(ALL_FUNCTIONS.items()))
def test_global_minimum_within_bounds(name, func):
    x_star, _ = func.global_minimum
    (lo1, hi1), (lo2, hi2) = func.bounds
    assert lo1 <= x_star[0] <= hi1
    assert lo2 <= x_star[1] <= hi2
