from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

from metaheuristics.algorithms.particle_swarm import ParticleSwarmOptimization
from metaheuristics.result import OptimizationResult

_ParamSpec = tuple[float, float] | tuple[float, float, str]


def _encode_bounds(param_space: dict[str, _ParamSpec]) -> tuple[list[str], list[tuple[float, float]]]:
    names = list(param_space)
    bounds = []
    for name in names:
        low, high = param_space[name][0], param_space[name][1]
        kind = param_space[name][2] if len(param_space[name]) > 2 else "linear"
        bounds.append((np.log10(low), np.log10(high)) if kind == "log" else (low, high))
    return names, bounds


def _decode(names: list[str], param_space: dict[str, _ParamSpec], x: np.ndarray) -> dict[str, float]:
    params = {}
    for name, value in zip(names, x):
        kind = param_space[name][2] if len(param_space[name]) > 2 else "linear"
        if kind == "log":
            value = 10**value
        elif kind == "int":
            value = int(round(value))
        params[name] = value
    return params


class MetaheuristicSearchCV(BaseEstimator):
    """Hyperparameter search driven by a metaheuristic optimizer.

    Minimizes $-\\text{CV score}$ over a hyperparameter box (sklearn scorers
    are always "higher is better", so this works for accuracy/F1/ROC-AUC as
    well as negative-oriented scorers like ``"neg_mean_squared_error"``), the
    same continuous-optimization framing used in
    ``notebooks/applications/hyperparameter_tuning.ipynb``. Any optimizer
    exposing ``optimize(objective_fn, bounds) -> OptimizationResult``
    (``GeneticAlgorithm``, ``ParticleSwarmOptimization``,
    ``DifferentialEvolution``, ``SlimeMouldAlgorithm``, ...) can be plugged in.

    Follows the scikit-learn search-CV convention (``best_params_``,
    ``best_score_``, ``best_estimator_``) so it's a drop-in alternative to
    ``GridSearchCV``/``RandomizedSearchCV``.

    Parameters
    ----------
    estimator : sklearn estimator
        Model whose hyperparameters are being tuned.
    param_space : dict[str, tuple]
        Maps hyperparameter name to ``(low, high)`` for a linear float,
        ``(low, high, "log")`` for a log10-scaled float, or
        ``(low, high, "int")`` for an integer.
    optimizer : object, default ``ParticleSwarmOptimization()``
        Metaheuristic optimizer instance exposing ``optimize(objective_fn, bounds)``.
    cv : int, default 5
        Number of cross-validation folds used to score each candidate.
    scoring : str or callable, optional
        Passed through to ``cross_val_score``; defaults to the estimator's score.
    refit : bool, default True
        Refit ``estimator`` with ``best_params_`` on the full data, enabling
        ``predict``/``score``.
    random_state : int, optional
        Seeds ``numpy.random`` before running the optimizer, for reproducibility.
    """

    def __init__(
        self,
        estimator: BaseEstimator = None,
        param_space: dict[str, _ParamSpec] = None,
        optimizer: object = None,
        cv: int = 5,
        scoring: str | Callable | None = None,
        refit: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.param_space = param_space
        self.optimizer = optimizer
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.random_state = random_state

    def fit(self, X, y):
        optimizer = self.optimizer if self.optimizer is not None else ParticleSwarmOptimization()
        names, bounds = _encode_bounds(self.param_space)

        def objective_fn(x: np.ndarray) -> float:
            params = _decode(names, self.param_space, x)
            estimator = clone(self.estimator).set_params(**params)
            return -cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring).mean()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        result: OptimizationResult = optimizer.optimize(objective_fn, bounds)

        self.best_params_ = _decode(names, self.param_space, result.best_solution)
        self.best_score_ = -result.best_fitness
        self.result_ = result

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.score(X, y)
