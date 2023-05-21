import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class ConstrainedLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept: bool = True, alpha: float = 0.0):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.alpha = alpha

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
        features_sign_constraints: dict = {},
        intercept_sign_constraint: int = 0,
        features_sum_constraint_equal: float = None,
    ) -> "ConstrainedLinearRegression":
        """
        Fits a linear model with constraints

        X : pandas.DataFrame of shape (n_samples, n_features)
            Training data.

        y : numpy.ndarray of shape (n_samples,)
            Target values.

        sample_weight : numpy.ndarray of shape (n_samples,), default=None
            Individual weights for each sample.

        features_sign_constraints : dict
            Dictionary with sign constraints. Keys must be from X's columns and values must take the values: -1, 0, 1
            indicating negative, unconstrained and positive sign respectively. Any column that is not present in the
            dictionary will default to 0.

        intercept_sign_constraint : int
            Indicates the sign of intercept, if present, and must take the values: -1, 0, 1

        features_sum_constraint_equal : float
            Constraints the sum of all coefficients plus intercept (if present)
        """
        X, y = check_X_y(X, y)
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

        features_sign_constraints_full = {feature: 0 for feature in range(n_features)}
        features_sign_constraints_full.update(features_sign_constraints)
        diag_values = list(features_sign_constraints_full.values())
        if self.fit_intercept:
            diag_values.append(intercept_sign_constraint)
        G = -1.0 * np.diag(diag_values)  # Negate since cvxopt by convention accepts inequalities of the form Gx <= h
        G = matrix(G)
        h = np.zeros(dim)
        h = matrix(h)

        A, b = None, None
        if features_sum_constraint_equal:
            A = np.ones(dim).astype("float")
            A = A.reshape(1, -1)
            A = matrix(A)
            b = np.array([features_sum_constraint_equal]).astype("float")
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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
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
