# Applications: Feature Selection & Hyperparameter Tuning

Implementation: [`src/metaheuristics/feature_selection.py`](../../src/metaheuristics/feature_selection.py),
[`src/metaheuristics/model_selection.py`](../../src/metaheuristics/model_selection.py)
Notebooks: [`feature_selection.ipynb`](../../notebooks/applications/feature_selection.ipynb),
[`feature_selection_vs_sklearn.ipynb`](../../notebooks/applications/feature_selection_vs_sklearn.ipynb),
[`hyperparameter_tuning.ipynb`](../../notebooks/applications/hyperparameter_tuning.ipynb),
[`hyperparameter_tuning_vs_sklearn.ipynb`](../../notebooks/applications/hyperparameter_tuning_vs_sklearn.ipynb)

## Idea

Feature selection and hyperparameter tuning are both naturally discrete/mixed search problems,
but each has a continuous relaxation that any of the five metaheuristics in this repo can drive
directly, via the shared `optimize(objective_fn, bounds) -> OptimizationResult` interface. Both
are then wrapped as scikit-learn estimators so they slot into existing `Pipeline`/`Search` code.

## Feature selection — `MetaheuristicSelector`

Relaxes "which subset of $N$ features?" into a continuous weight $w_i \in [0, 1]$ per feature,
thresholded at $0.5$ to recover a binary mask — same trick used across the notebook (see
`docs/notes` for the individual algorithms that can plug in here).

$$f(w) = \underbrace{-\text{score}(\text{mask}(w))}_{\text{fidelity}} + \underbrace{\alpha \cdot \|\text{mask}(w)\|_0 / N}_{\text{sparsity}}, \quad \text{mask}(w)_i = \mathbb{1}[w_i > 0.5]$$

`score` is `cross_val_score(estimator, X[:, mask], y, cv=cv, scoring=scoring).mean()`; a mask that
selects zero features returns $+\infty$ (an invalid subset, not just a bad score).

Implements `SelectorMixin` + `BaseEstimator` (`fit`/`transform`/`get_support`), so it drops into a
`Pipeline` next to `RFE` or `SelectKBest`.

| Parameter | Default | Meaning |
|---|---|---|
| `estimator` | `SVC()` | Model used to score candidate feature subsets |
| `optimizer` | `ParticleSwarmOptimization()` | Any optimizer exposing `optimize(objective_fn, bounds)` |
| `alpha` | `0.02` | Weight on the feature-count penalty |
| `cv` | `5` | Cross-validation folds |
| `scoring` | estimator's own `.score()` | Any sklearn scorer name/callable, incl. negative-oriented ones (e.g. `"neg_mean_squared_error"`) — sklearn scorers are always "higher is better", so the sign convention just carries through `-score` |
| `random_state` | `None` | Seeds `numpy.random` before the optimizer runs |

Fitted attributes: `weights_`, `mask_`, `n_features_in_`, and `result_` (the raw
`OptimizationResult`, for convergence plots).

```python
from sklearn.linear_model import LogisticRegression
from metaheuristics.feature_selection import MetaheuristicSelector

selector = MetaheuristicSelector(estimator=LogisticRegression(), alpha=0.02, cv=5, random_state=0)
X_selected = selector.fit_transform(X, y)
selector.mask_, selector.get_support()  # boolean feature mask
```

`alpha` is scaled assuming `scoring` is roughly `[0, 1]`-bounded (accuracy, F1, ROC-AUC, R²); for
an unbounded scorer, rescale `alpha` so the sparsity term doesn't become negligible (or dominant).

## Hyperparameter tuning — `MetaheuristicSearchCV`

Relaxes a hyperparameter search space into a continuous box, one dimension per hyperparameter,
via a `param_space` spec — a drop-in alternative to `GridSearchCV`/`RandomizedSearchCV`.

`param_space` entries, per hyperparameter name:

| Spec | Encoding | Use for |
|---|---|---|
| `(low, high)` | linear float | e.g. `dropout: (0.0, 0.5)` |
| `(low, high, "log")` | $\log_{10}$-scaled float, decoded via $10^x$ | e.g. `C: (1e-2, 1e4, "log")` |
| `(low, high, "int")` | linear, decoded via `round(x)` | e.g. `n_estimators: (10, 500, "int")` |

Objective: $f(x) = -\text{cross\_val\_score}(\text{estimator.set\_params}(\text{decode}(x)))$ — same
"higher is better" convention as `MetaheuristicSelector`, so any sklearn scorer works.

| Parameter | Default | Meaning |
|---|---|---|
| `estimator` | *(required)* | Model whose hyperparameters are tuned |
| `param_space` | *(required)* | `dict[str, spec]`, see table above |
| `optimizer` | `ParticleSwarmOptimization()` | Any optimizer exposing `optimize(objective_fn, bounds)` |
| `cv` | `5` | Cross-validation folds |
| `scoring` | estimator's own `.score()` | Any sklearn scorer name/callable |
| `refit` | `True` | Refit on full data with `best_params_`, enabling `predict`/`score` |
| `random_state` | `None` | Seeds `numpy.random` before the optimizer runs |

Fitted attributes: `best_params_`, `best_score_`, `best_estimator_` (if `refit=True`), and
`result_` (the raw `OptimizationResult`).

```python
from sklearn.svm import SVC
from metaheuristics.model_selection import MetaheuristicSearchCV

search = MetaheuristicSearchCV(
    estimator=SVC(),
    param_space={"C": (1e-2, 1e4, "log"), "gamma": (1e-6, 1e1, "log")},
    cv=5,
    random_state=0,
)
search.fit(X, y)
search.best_params_, search.best_score_
```

## When to reach for these vs. `GridSearchCV`/`RandomizedSearchCV`

`GridSearchCV` exhaustively evaluates a fixed lattice — simple, but resolution is fixed up front
and most of the grid is usually far from the optimum. `RandomizedSearchCV` samples independently,
so it never gets sharper as evidence accumulates. The metaheuristic wrappers instead steer each
new batch of candidates using every CV score observed so far, which tends to reach a given
accuracy in fewer evaluations (see the "anytime performance" plot in
`hyperparameter_tuning_vs_sklearn.ipynb`) — at the cost of an optimizer with its own
hyperparameters (population size, iteration count) to set, and no built-in `n_jobs` parallelism or
full `cv_results_` grid like sklearn's searchers provide.
