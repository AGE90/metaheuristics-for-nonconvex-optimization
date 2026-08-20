from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from metaheuristics.algorithms.particle_swarm import ParticleSwarmOptimization
from metaheuristics.model_selection import MetaheuristicSearchCV

X, y = make_classification(
    n_samples=100, n_features=10, n_informative=4, n_redundant=0, random_state=0
)

PARAM_SPACE = {"C": (1e-2, 1e4, "log"), "gamma": (1e-6, 1e1, "log")}


def make_search():
    return MetaheuristicSearchCV(
        estimator=SVC(),
        param_space=PARAM_SPACE,
        optimizer=ParticleSwarmOptimization(num_particles=8, max_iterations=5),
        cv=3,
        random_state=0,
    )


def test_fit_sets_best_params_and_score():
    search = make_search()
    search.fit(X, y)

    assert set(search.best_params_) == set(PARAM_SPACE)
    for name, (low, high, *_rest) in PARAM_SPACE.items():
        assert low <= search.best_params_[name] <= high
    assert 0.0 <= search.best_score_ <= 1.0


def test_cloneable():
    search = make_search()
    cloned = clone(search)
    assert cloned.get_params()["cv"] == 3
    assert not hasattr(cloned, "best_params_")  # clone() must not carry fitted state


def test_predict_and_score_after_refit():
    search = make_search()
    search.fit(X, y)

    predictions = search.predict(X)
    assert predictions.shape == (100,)
    assert 0.0 <= search.score(X, y) <= 1.0
