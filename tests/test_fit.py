import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
import pytest
from constrainedlr.model import ConstrainedLinearRegression

atol = 1e-5

dataset = load_diabetes()
X = dataset["data"]
y = dataset["target"]


def test_no_intercept():
    clr = ConstrainedLinearRegression(fit_intercept=False)
    clr.fit(X, y)
    assert clr.intercept_ is None
    assert clr.coef_.shape[0] == X.shape[1]


def test_intercept():
    clr = ConstrainedLinearRegression(fit_intercept=True)
    clr.fit(X, y)
    assert clr.intercept_ is not None
    assert clr.coef_.shape[0] == X.shape[1]


def test_unconstrained():
    clr = ConstrainedLinearRegression(fit_intercept=True)
    clr.fit(X, y)

    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)

    assert np.allclose(lr.intercept_, clr.intercept_, atol=atol)
    assert np.allclose(lr.coef_, clr.coef_, atol=atol)


def test_all_positive():
    clr = ConstrainedLinearRegression(fit_intercept=True)
    clr.fit(X, y, coefficients_sign_constraints={col: 1 for col in range(X.shape[1])})

    lr = LinearRegression(fit_intercept=True, positive=True)
    lr.fit(X, y)

    assert np.allclose(lr.intercept_, clr.intercept_, atol=atol)
    assert np.allclose(lr.coef_, clr.coef_, atol=atol)


def test_feature_signs():
    clr = ConstrainedLinearRegression(fit_intercept=True)

    # Perform multiple tests since signs are produced randomly
    np.random.seed(0)
    for _ in range(30):
        signs = random.choices([-1, 1, "positive", "negative"], k=X.shape[1])
        features_sign_constraints = dict(zip(list(range(X.shape[1])), signs))
        clr.fit(X, y, coefficients_sign_constraints=features_sign_constraints)

        # if coefficients multipled with imposed signs are all positive (or approximately positive) then pass the test
        signs_numeric = np.array([1 if s == "positive" else -1 if s == "negative" else s for s in signs])
        assert np.all(clr.coef_ * signs_numeric > -atol)


def test_intercept_sign():
    clr = ConstrainedLinearRegression(fit_intercept=True)
    clr.fit(X, y, intercept_sign_constraint=1)
    assert clr.intercept_ > 0

    clr.fit(X, y, intercept_sign_constraint=-1)
    assert clr.intercept_ < 0

    clr.fit(X, y, intercept_sign_constraint="positive")
    assert clr.intercept_ > 0

    clr.fit(X, y, intercept_sign_constraint="negative")
    assert clr.intercept_ < 0

    # Check the below runs without raising any exceptions
    try:
        clr.fit(X, y, intercept_sign_constraint=0)
        clr.fit(X, y)
    except Exception as exception:
        assert False

    # Check if exception is raised when an invalid value is given
    with pytest.raises(ValueError):
        clr.fit(X, y, intercept_sign_constraint="invalid value")
    with pytest.raises(ValueError):
        clr.fit(X, y, intercept_sign_constraint=2)


def test_features_sum():
    clr = ConstrainedLinearRegression(fit_intercept=True)
    features_sum_constraint_equal = 15
    clr.fit(X, y, coefficients_sum_constraint=features_sum_constraint_equal)
    sum_of_weights = clr.coef_.sum() + clr.intercept_
    assert np.allclose(sum_of_weights, features_sum_constraint_equal, atol=atol)

    clr = ConstrainedLinearRegression(fit_intercept=False)
    features_sum_constraint_equal = 15
    clr.fit(X, y, coefficients_sum_constraint=features_sum_constraint_equal)
    sum_of_weights = clr.coef_.sum()
    assert np.allclose(sum_of_weights, features_sum_constraint_equal, atol=atol)


def test_alpha():
    clr = ConstrainedLinearRegression(fit_intercept=True, alpha=1.0)
    clr.fit(X, y)

    ridge = Ridge(fit_intercept=True, alpha=1.0)
    ridge.fit(X, y)

    assert np.allclose(ridge.intercept_, clr.intercept_, rtol=0.01)
    assert np.allclose(ridge.coef_, clr.coef_, rtol=0.01)


def test_sample_weight():
    # Perform multiple tests since sample weights are produced randomly
    np.random.seed(0)
    for _ in range(10):
        sample_weight = np.random.random(X.shape[0]) * 10

        clr = ConstrainedLinearRegression()
        clr.fit(X, y, sample_weight=sample_weight)

        lr = LinearRegression()
        lr.fit(X, y, sample_weight=sample_weight)

        assert np.allclose(lr.intercept_, clr.intercept_, atol=atol)
        assert np.allclose(lr.coef_, clr.coef_, atol=atol)


def test_coefficients_range_constraints():
    clr = ConstrainedLinearRegression(fit_intercept=True)

    # Perform multiple tests since bounds are produced randomly
    np.random.seed(0)
    for _ in range(50):
        n_contraints = np.random.randint(0, X.shape[1])

        # Add range constraints in a random set of features
        coefficients_range_constraints = {}
        for _ in range(n_contraints):
            feature = np.random.randint(0, X.shape[1])

            # Add either lower, or upper, or both contraints
            feature_contraints = {}
            if np.random.random() > 0.5:
                feature_contraints.update({"upper": np.random.normal(100, 20)})
            if np.random.random() > 0.5:
                feature_contraints.update({"lower": np.random.normal(-100, 20)})
            coefficients_range_constraints.update({feature: feature_contraints})

        clr.fit(X, y, coefficients_range_constraints=coefficients_range_constraints)

        for feature, contraints in coefficients_range_constraints.items():
            if "upper" in contraints:
                assert clr.coef_[feature] <= contraints["upper"] + atol
            if "lower" in contraints:
                assert clr.coef_[feature] >= contraints["lower"] - atol
