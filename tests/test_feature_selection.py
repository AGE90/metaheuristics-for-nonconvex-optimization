import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from metaheuristics.algorithms.particle_swarm import ParticleSwarmOptimization
from metaheuristics.feature_selection import MetaheuristicSelector

X, y = make_classification(
    n_samples=100, n_features=10, n_informative=4, n_redundant=0, random_state=0
)


def make_selector():
    return MetaheuristicSelector(
        estimator=LogisticRegression(),
        optimizer=ParticleSwarmOptimization(num_particles=8, max_iterations=5),
        cv=3,
        random_state=0,
    )


def test_fit_transform_selects_subset():
    selector = make_selector()
    X_selected = selector.fit_transform(X, y)

    assert selector.mask_.shape == (10,)
    assert selector.n_features_in_ == 10
    assert 0 < selector.mask_.sum() <= 10
    assert X_selected.shape == (100, selector.mask_.sum())
    np.testing.assert_array_equal(selector.get_support(), selector.mask_)


def test_cloneable():
    selector = make_selector()
    cloned = clone(selector)
    assert cloned.get_params()["cv"] == 3
    assert not hasattr(cloned, "mask_")  # clone() must not carry fitted state


def test_works_in_pipeline():
    pipeline = Pipeline([
        ("select", make_selector()),
        ("classify", LogisticRegression()),
    ])
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert predictions.shape == (100,)
