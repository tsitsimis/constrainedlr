import random

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from constrainedlr.model import ConstrainedLinearRegression
from constrainedlr.validation import convert_feature_names_to_indices

atol = 1e-5

dataset = load_diabetes()
X = dataset["data"]
y = dataset["target"]

X_df = pd.DataFrame(X, columns=dataset["feature_names"])


def test_no_intercept() -> None:
    clr = ConstrainedLinearRegression(fit_intercept=False)
    clr.fit(X, y)
    y_pred = clr.predict(X)
    assert clr.intercept_ is None
    assert clr.coef_.shape[0] == X.shape[1]
    assert y_pred.shape[0] == X.shape[0]


def test_intercept() -> None:
    clr = ConstrainedLinearRegression(fit_intercept=True)
    clr.fit(X, y)
    y_pred = clr.predict(X)
    assert clr.intercept_ is not None
    assert clr.coef_.shape[0] == X.shape[1]
    assert y_pred.shape[0] == X.shape[0]


def test_pandas_input() -> None:
    clr = ConstrainedLinearRegression(fit_intercept=True)

    df = pd.DataFrame(X, columns=[f"col{i}" for i in range(X.shape[1])])
    clr.fit(df, y)
    y_pred = clr.predict(df)

    assert clr.intercept_ is not None
    assert clr.coef_.shape[0] == X.shape[1]
    assert y_pred.shape[0] == X.shape[0]


def test_pipeline() -> None:
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("reg", ConstrainedLinearRegression(fit_intercept=True, alpha=1.0))]
    )
    pipeline.fit(
        X,
        y,
        reg__coefficients_range_constraints={6: {"lower": 1}, 7: {"lower": 0, "upper": 1}},
        reg__coefficients_sum_constraint=10,
        reg__intercept_sign_constraint="negative",
    )
    y_pred = pipeline.predict(X)

    assert pipeline.named_steps["reg"].intercept_ is not None
    assert pipeline.named_steps["reg"].coef_.shape[0] == X.shape[1]
    assert y_pred.shape[0] == X.shape[0]


def test_pipeline_named_features_sign() -> None:
    sklearn.set_config(transform_output="pandas")

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("reg", ConstrainedLinearRegression(fit_intercept=True, alpha=1.0))]
    )
    pipeline.fit(
        X_df,
        y,
        reg__coefficients_sign_constraints={"s1": "positive", "s3": "negative"},
        reg__coefficients_sum_constraint=10,
        reg__intercept_sign_constraint="negative",
    )
    y_pred = pipeline.predict(X_df)

    assert pipeline.named_steps["reg"].intercept_ is not None
    assert pipeline.named_steps["reg"].coef_.shape[0] == X.shape[1]
    assert y_pred.shape[0] == X.shape[0]


def test_pipeline_named_features_range() -> None:
    sklearn.set_config(transform_output="pandas")

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("reg", ConstrainedLinearRegression(fit_intercept=True, alpha=1.0))]
    )
    pipeline.fit(
        X_df,
        y,
        reg__coefficients_range_constraints={"s1": {"lower": 1}, "s3": {"lower": 0, "upper": 1}},
        reg__coefficients_sum_constraint=10,
        reg__intercept_sign_constraint="negative",
    )
    y_pred = pipeline.predict(X_df)

    assert pipeline.named_steps["reg"].intercept_ is not None
    assert pipeline.named_steps["reg"].coef_.shape[0] == X.shape[1]
    assert y_pred.shape[0] == X.shape[0]


def test_unconstrained() -> None:
    clr = ConstrainedLinearRegression(fit_intercept=True)
    clr.fit(X, y)
    y_crl_pred = clr.predict(X)

    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)
    y_lr_pred = clr.predict(X)

    assert np.allclose(lr.intercept_, clr.intercept_, atol=atol)
    assert np.allclose(lr.coef_, clr.coef_, atol=atol)
    assert np.allclose(y_lr_pred, y_crl_pred, atol=atol)


def test_all_positive() -> None:
    clr = ConstrainedLinearRegression(fit_intercept=True)
    clr.fit(X, y, coefficients_sign_constraints={col: 1 for col in range(X.shape[1])})

    lr = LinearRegression(fit_intercept=True, positive=True)
    lr.fit(X, y)

    assert np.allclose(lr.intercept_, clr.intercept_, atol=atol)
    assert np.allclose(lr.coef_, clr.coef_, atol=atol)


def test_all_positive_named_features() -> None:
    clr = ConstrainedLinearRegression(fit_intercept=True)
    clr.fit(X_df, y, coefficients_sign_constraints={col: 1 for col in X_df.columns})

    lr = LinearRegression(fit_intercept=True, positive=True)
    lr.fit(X_df, y)

    assert np.allclose(lr.intercept_, clr.intercept_, atol=atol)
    assert np.allclose(lr.coef_, clr.coef_, atol=atol)


def test_feature_signs() -> None:
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


def test_intercept_sign() -> None:
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
    except Exception:
        raise AssertionError from None

    # Check if exception is raised when an invalid value is given
    with pytest.raises(ValueError):
        clr.fit(X, y, intercept_sign_constraint="invalid value")
    with pytest.raises(ValueError):
        clr.fit(X, y, intercept_sign_constraint=2)


def test_features_sum() -> None:
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


def test_alpha() -> None:
    clr = ConstrainedLinearRegression(fit_intercept=True, alpha=1.0)
    clr.fit(X, y)

    ridge = Ridge(fit_intercept=True, alpha=1.0)
    ridge.fit(X, y)

    assert np.allclose(ridge.intercept_, clr.intercept_, rtol=0.01)
    assert np.allclose(ridge.coef_, clr.coef_, rtol=0.01)


def test_sample_weight() -> None:
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


def test_coefficients_range_constraints() -> None:
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


def test_mixed_constraints() -> None:
    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(
            X,
            y,
            coefficients_sign_constraints={0: "positive"},
            coefficients_range_constraints={0: {"lower": 0, "upper": 1}},
        )


def test_non_dict_constraints_sign_constraints() -> None:
    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_sign_constraints=100)


def test_non_int_indices_sign_constraints() -> None:
    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_sign_constraints={"de": 1})

    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_sign_constraints={1.2: 1})

    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_sign_constraints={-1: 1})

    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_sign_constraints={X.shape[0]: 1})


def test_invalid_sign_sign_constraints() -> None:
    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_sign_constraints={0: 3})

    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_sign_constraints={0: "very positive"})

    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_sign_constraints={0: -2})


def test_non_dict_constraints_range_constraints() -> None:
    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_range_constraints=100)


def test_non_int_indices_range_constraints() -> None:
    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_range_constraints={"de": 1})

    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_range_constraints={1.2: 1})

    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_range_constraints={-1: 1})

    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_range_constraints={X.shape[0]: 1})


def test_invalid_range_constraints() -> None:
    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_range_constraints={0: {"bottom": 0}})

    with pytest.raises(ValueError):
        clr = ConstrainedLinearRegression()
        clr.fit(X, y, coefficients_range_constraints={0: {"lower": 1, "upper": 0}})


def test_convert_feature_names_in_sign_constraints_to_indices():
    constraints = {"age": 0, "bmi": "negative"}
    feature_names_in_ = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    formatted_constraints = convert_feature_names_to_indices(constraints, feature_names_in_)
    assert formatted_constraints == {0: 0, 2: "negative"}


def test_convert_feature_names_in_sign_constraints_to_indices_pipeline():
    constraints = {"age": 0, "bmi": "negative"}
    feature_names_in_ = [
        "standardscaler-1__age",
        "standardscaler-1__sex",
        "standardscaler-1__bmi",
        "standardscaler-1__bp",
        "standardscaler-1__s1",
        "standardscaler-1__s2",
        "standardscaler-1__s3",
        "standardscaler-1__s4",
        "standardscaler-1__s5",
        "standardscaler-1__s6",
    ]
    formatted_constraints = convert_feature_names_to_indices(constraints, feature_names_in_)
    assert formatted_constraints == {0: 0, 2: "negative"}


def test_convert_feature_names_in_range_constraints_to_indices():
    constraints = {"age": {"lower": 0, "upper": 3}, "bmi": {"upper": -1}}
    feature_names_in_ = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    formatted_constraints = convert_feature_names_to_indices(constraints, feature_names_in_)
    assert formatted_constraints == {0: {"lower": 0, "upper": 3}, 2: {"upper": -1}}


def test_convert_feature_names_in_range_constraints_to_indices_pipeline():
    constraints = {"age": {"lower": 0, "upper": 3}, "bmi": {"upper": -1}}
    feature_names_in_ = [
        "standardscaler-1__age",
        "standardscaler-1__sex",
        "standardscaler-1__bmi",
        "standardscaler-1__bp",
        "standardscaler-1__s1",
        "standardscaler-1__s2",
        "standardscaler-1__s3",
        "standardscaler-1__s4",
        "standardscaler-1__s5",
        "standardscaler-1__s6",
    ]
    formatted_constraints = convert_feature_names_to_indices(constraints, feature_names_in_)
    assert formatted_constraints == {0: {"lower": 0, "upper": 3}, 2: {"upper": -1}}
