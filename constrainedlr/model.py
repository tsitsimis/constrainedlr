"""
Constrained Linear Model
"""

# ruff: noqa: N806 (non-lower-case-variable-in-function)
# ruff: noqa: N803 (invalid-argument-name)

from typing import Optional, Union

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from constrainedlr.validation import (
    validate_coefficients_range_constraints,
    validate_coefficients_sign_constraints,
    validate_intercept_sign_constraint,
)


class ConstrainedLinearRegression(BaseEstimator, RegressorMixin):
    """
    Least squares Linear Regression with optional constraints on its coefficients/weights.

    ConstrainedLinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual
    sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation,
    while at the same time imposing constraints on the signs and values of the coefficients.
    """

    def __init__(self, fit_intercept: bool = True, alpha: float = 0.0):
        """
        ConstrainedLinearRegression constructor

        Args:
            fit_intercept:
                Whether to calculate the intercept for this model.

            alpha:
                Constant that multiplies the L2 penalty term, controlling regularization strength.
                alpha must be a non-negative float i.e. in [0, inf).

        Attributes:
            coef_:
                Weight vector of shape (n_features,).

            intercept_:
                Independent/constant term in regression model. Set to None if fit_intercept = False.
        """
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.alpha = alpha
        self.feature_names_in_: Optional[np.ndarray[str]] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],  # noqa: N803
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        coefficients_sign_constraints: Optional[dict] = None,
        coefficients_range_constraints: Optional[dict] = None,
        intercept_sign_constraint: Union[int, str] = 0,
        coefficients_sum_constraint: Optional[float] = None,
    ) -> "ConstrainedLinearRegression":
        """
        Fit linear model with constraints.

        Args:
            X:
                Training data of shape (n_samples, n_features).

            y:
                Target values of shape (n_samples,).

            sample_weight:
                Individual weights of shape (n_samples,) for each sample.

            coefficients_sign_constraints:
                Dictionary with sign constraints. Keys must be integers specifying the location of the corresponding
                feature in the columns in the dataset. Values must take the values: -1, 0, 1 indicating negative,
                unconstrained and positive sign respectively. Any column that is not present in the
                dictionary will default to 0.

            coefficients_range_constraints:
                Dictionary of the form: `{column_index: {"lower": <float>, "upper": <float>}}`.
                Eiter both or one of lower or upper bounds can be specified. If a column index is not specified,
                the coefficient remains unconstrained. Only one of `features_sign_constraints`
                or `coefficients_range_constraints` can be provided.

            intercept_sign_constraint:
                Indicates the sign of intercept, if present, and must take the values: -1, 0, 1.

            coefficients_sum_constraint:
                Constraints the sum of all coefficients plus intercept (if present).

        Returns:
            Fitted Estimator.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns.to_list())
            print(self.feature_names_in_)

        coefficients_sign_constraints = validate_coefficients_sign_constraints(
            coefficients_sign_constraints, X, self.feature_names_in_
        )
        intercept_sign_constraint = validate_intercept_sign_constraint(intercept_sign_constraint)
        coefficients_range_constraints = validate_coefficients_range_constraints(
            coefficients_range_constraints, X, self.feature_names_in_
        )

        X, y = check_X_y(X, y)

        if len(coefficients_sign_constraints) > 0 and len(coefficients_range_constraints) > 0:
            raise ValueError(
                "Only one of `coefficients_sign_constraints` or `coefficients_range_constraints` can be provided."
            )

        n_samples, n_features = X.shape

        # Augment features to fit intercept
        if self.fit_intercept:
            X = np.hstack([X, np.ones(n_samples).reshape(-1, 1)])

        dim = X.shape[1]

        # Weight matrix
        W = np.eye(n_samples) if sample_weight is None else np.diag(sample_weight)

        # Quadratic program
        P = X.T.dot(W).dot(X) + self.alpha * np.eye(dim)
        P = matrix(P)
        q = (-y.T.dot(W).dot(X)).T
        q = matrix(q)

        G, h = None, None
        features_sign_constraints_full = {feature: 0 for feature in range(n_features)}
        features_sign_constraints_full.update(coefficients_sign_constraints)
        diag_values = list(features_sign_constraints_full.values())
        if self.fit_intercept:
            diag_values.append(intercept_sign_constraint)
        G = -1.0 * np.diag(diag_values)  # Negate since cvxopt by convention accepts inequalities of the form Gx <= h
        G = matrix(G)
        h = np.zeros(dim)
        h = matrix(h)

        if len(coefficients_range_constraints) > 0:
            coefficients_upper_bound_constraints = {
                k: v for k, v in coefficients_range_constraints.items() if "upper" in v
            }
            G_upper = np.zeros((len(coefficients_upper_bound_constraints), dim))
            for i, feature in enumerate(coefficients_upper_bound_constraints.keys()):
                G_upper[i, feature] = 1
            h_upper = np.array([v["upper"] for k, v in coefficients_upper_bound_constraints.items()])

            coefficients_lower_bound_constraints = {
                k: v for k, v in coefficients_range_constraints.items() if "lower" in v
            }
            G_lower = np.zeros((len(coefficients_lower_bound_constraints), dim))
            for i, feature in enumerate(coefficients_lower_bound_constraints.keys()):
                G_lower[i, feature] = -1
            h_lower = -1.0 * np.array([v["lower"] for k, v in coefficients_lower_bound_constraints.items()])

            G = np.concatenate([G_upper, G_lower], axis=0).astype("float")
            G = matrix(G)
            h = np.concatenate([h_upper, h_lower])
            h = matrix(h)

        A, b = None, None
        if coefficients_sum_constraint:
            A = np.ones(dim).astype("float")
            A = A.reshape(1, -1)
            A = matrix(A)
            b = np.array([coefficients_sum_constraint]).astype("float")
            b = matrix(b)

        solvers.options["show_progress"] = False
        solver = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
        weights = np.array(solver["x"]).flatten()

        if self.fit_intercept:
            self.coef_ = weights[0:-1]
            self.intercept_ = weights[-1]
        else:
            self.coef_ = weights

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict using the linear model.

        Parameters:
            X:
                Samples of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples,).
        """
        check_is_fitted(self)
        X = check_array(X)

        n_samples = X.shape[0]

        # Augment features for intercept
        if self.fit_intercept:
            X = np.hstack([X, np.ones(n_samples).reshape(-1, 1)])
            weights = np.concatenate([self.coef_, [self.intercept_]])
        else:
            weights = self.coef_  # type: ignore

        y_pred = X.dot(weights)
        return y_pred  # type: ignore

    def get_feature_names_out(self) -> np.ndarray[str]:
        """Get output feature names

        Returns:
            List of feature names
        """
        return self.feature_names_in_
