from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted, check_X_y

from metaheuristics.algorithms.particle_swarm import ParticleSwarmOptimization
from metaheuristics.result import OptimizationResult


class MetaheuristicSelector(SelectorMixin, BaseEstimator):
    """Wrapper feature selector driven by a metaheuristic optimizer.

    Selects a feature subset by minimizing cross-validated classification
    error (plus a penalty on subset size) over continuous weights
    $w_i \\in [0, 1]$, thresholded at $0.5$ — the same continuous-relaxation
    trick used in ``notebooks/applications/feature_selection.ipynb``. Any
    optimizer exposing ``optimize(objective_fn, bounds) -> OptimizationResult``
    (``GeneticAlgorithm``, ``ParticleSwarmOptimization``,
    ``DifferentialEvolution``, ``SlimeMouldAlgorithm``, ...) can be plugged in.

    Implements the scikit-learn transformer API (``fit``/``transform``/
    ``get_support``), so it can be used in a ``Pipeline`` alongside
    ``sklearn.feature_selection`` selectors like ``RFE`` or ``SelectKBest``.

    Parameters
    ----------
    estimator : sklearn estimator, default ``SVC()``
        Model used to score candidate feature subsets via cross-validation.
    optimizer : object, default ``ParticleSwarmOptimization()``
        Metaheuristic optimizer instance exposing ``optimize(objective_fn, bounds)``.
    alpha : float, default 0.02
        Weight on the feature-count penalty (fraction of features used).
    cv : int, default 5
        Number of cross-validation folds used to score each candidate subset.
    scoring : str or callable, optional
        Passed through to ``cross_val_score``; defaults to the estimator's score.
        Any scikit-learn scorer works, including negative-oriented ones like
        ``"neg_mean_squared_error"`` — sklearn scorers are always "higher is
        better", and the objective minimizes ``-score``, so the sign convention
        just carries through. The ``alpha`` penalty is on a fixed ``[0, 1]``-ish
        scale, so for unbounded scorers you may need to rescale ``alpha`` to
        keep the sparsity term from becoming negligible (or dominant).
    random_state : int, optional
        Seeds ``numpy.random`` before running the optimizer, for reproducibility.
    """

    def __init__(
        self,
        estimator: BaseEstimator = None,
        optimizer: object = None,
        alpha: float = 0.02,
        cv: int = 5,
        scoring: str | Callable | None = None,
        random_state: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.optimizer = optimizer
        self.alpha = alpha
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        num_features = X.shape[1]
        estimator = clone(self.estimator) if self.estimator is not None else SVC()
        optimizer = self.optimizer if self.optimizer is not None else ParticleSwarmOptimization()

        def objective_fn(weights: np.ndarray) -> float:
            mask = weights > 0.5
            if not mask.any():
                return np.inf
            score = cross_val_score(estimator, X[:, mask], y, cv=self.cv, scoring=self.scoring).mean()
            return -score + self.alpha * mask.sum() / num_features

        if self.random_state is not None:
            np.random.seed(self.random_state)

        result: OptimizationResult = optimizer.optimize(objective_fn, [(0.0, 1.0)] * num_features)

        self.weights_ = result.best_solution
        self.mask_ = result.best_solution > 0.5
        self.result_ = result
        self.n_features_in_ = num_features
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "mask_")
        return self.mask_
