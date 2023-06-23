from typing import Union

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .validation import validate_coefficients_sign_constraints, validate_coefficients_range_constraints


class ConstrainedLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept: bool = True, alpha: float = 0.0):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.alpha = alpha

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        sample_weight: np.ndarray = None,
        coefficients_sign_constraints: dict = {},
        coefficients_range_constraints: dict = {},
        intercept_sign_constraint: int = 0,
        coefficients_sum_constraint: float = None,
    ) -> "ConstrainedLinearRegression":
        """
        Fits a linear model with constraints

        X : {numpy.ndarray, pandas.DataFrame} of shape (n_samples, n_features)
            Training data.

        y : numpy.ndarray of shape (n_samples,)
            Target values.

        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Individual weights for each sample.

        features_sign_constraints : dict
            Dictionary with sign constraints. Keys must be integers specifying the location of the corresponding feature
            in the columns in the dataset. Values must take the values: -1, 0, 1 indicating negative,
            unconstrained and positive sign respectively. Any column that is not present in the
            dictionary will default to 0.

        coefficients_range_constraints : dict
            Dictionary of the form: `{column_index: {"lower": <float>, "upper": <float>}}`.
            Eiter both or one of lower or upper bounds can be specified. If a column index is not specified,
            the coefficient remains unconstrained. Only one of `features_sign_constraints` or `coefficients_range_constraints`
            can be provided.

        intercept_sign_constraint : int
            Indicates the sign of intercept, if present, and must take the values: -1, 0, 1

        features_sum_constraint_equal : float
            Constraints the sum of all coefficients plus intercept (if present)
        """
        X, y = check_X_y(X, y)
        validate_coefficients_sign_constraints(coefficients_sign_constraints, X)
        validate_coefficients_range_constraints(coefficients_range_constraints, X)

        if len(coefficients_sign_constraints) > 0 and len(coefficients_range_constraints) > 0:
            raise ValueError(
                "Only one of `features_sign_constraints` or `coefficients_range_constraints` can be provided."
            )

        if np.ndim(y) == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Augment features to fit intercept
        if self.fit_intercept:
            X = np.hstack([X, np.ones(n_samples).reshape(-1, 1)])

        dim = X.shape[1]

        # Weight matrix
        if sample_weight is None:
            W = np.eye(n_samples)
        else:
            W = np.diag(sample_weight)

        # Quadratic program
        P = X.T.dot(W).dot(X) + self.alpha * np.eye(dim)
        P = matrix(P)
        q = (-y.T.dot(W).dot(X)).T
        q = matrix(q)

        G, h = None, None
        if len(coefficients_sign_constraints) > 0:
            features_sign_constraints_full = {feature: 0 for feature in range(n_features)}
            features_sign_constraints_full.update(coefficients_sign_constraints)
            diag_values = list(features_sign_constraints_full.values())
            if self.fit_intercept:
                diag_values.append(intercept_sign_constraint)
            G = -1.0 * np.diag(
                diag_values
            )  # Negate since cvxopt by convention accepts inequalities of the form Gx <= h
            G = matrix(G)
            h = np.zeros(dim)
            h = matrix(h)
        elif len(coefficients_range_constraints) > 0:
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
        check_is_fitted(self)
        X = check_array(X)

        n_samples = X.shape[0]

        # Augment features for intercept
        if self.fit_intercept:
            X = np.hstack([X, np.ones(n_samples).reshape(-1, 1)])
            weights = np.concatenate([self.coef_, [self.intercept_]])
        else:
            weights = self.coef_

        y_pred = X.dot(weights)
        return y_pred
