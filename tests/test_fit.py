import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge

from constrainedlr.model import ConstrainedLinearRegression

atol = 1e-5

dataset = load_diabetes()
X = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
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
    clr.fit(X, y, features_sign_constraints={col: 1 for col in X.columns})

    lr = LinearRegression(fit_intercept=True, positive=True)
    lr.fit(X, y)

    assert np.allclose(lr.intercept_, clr.intercept_, atol=atol)
    assert np.allclose(lr.coef_, clr.coef_, atol=atol)


def test_feature_signs():
    clr = ConstrainedLinearRegression(fit_intercept=True)

    # Perform multiple tests since signs are produced randomly
    np.random.seed(0)
    for _ in range(10):
        signs = np.random.choice([-1, 1], size=X.shape[1])
        features_sign_constraints = dict(zip(X.columns, signs))
        clr.fit(X, y, features_sign_constraints=features_sign_constraints)

        assert np.alltrue(np.sign(clr.coef_) == signs)


def test_intercept_sign():
    clr = ConstrainedLinearRegression(fit_intercept=True)
    clr.fit(X, y, intercept_sign_constraint=1)
    assert clr.intercept_ > 0

    clr.fit(X, y, intercept_sign_constraint=-1)
    assert clr.intercept_ < 0


def test_features_sum():
    clr = ConstrainedLinearRegression(fit_intercept=True)
    features_sum_constraint_equal = 15
    clr.fit(X, y, features_sum_constraint_equal=features_sum_constraint_equal)
    sum_of_weights = clr.coef_.sum() + clr.intercept_
    assert np.allclose(sum_of_weights, features_sum_constraint_equal, atol=atol)

    clr = ConstrainedLinearRegression(fit_intercept=False)
    features_sum_constraint_equal = 15
    clr.fit(X, y, features_sum_constraint_equal=features_sum_constraint_equal)
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
