from typing import Union

import numpy as np
import osqp
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin


class ConstrainedLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series], constraints) -> "ConstrainedLinearRegression":
        X_ = X.values.copy()

        if type(y) == pd.Series:
            y_ = y.values.copy()
        else:
            y_ = y.copy()

        if np.ndim(y_) == 1:
            y_ = y_.reshape(-1, 1)

        n_samples = X.shape[0]

        # Augment features to fit intercept
        if self.fit_intercept:
            X_ = np.hstack([X_, np.ones(n_samples).reshape(-1, 1)])

        dim = X_.shape[1]

        P = X_.T.dot(X_)
        P = sparse.csc_matrix(P)
        q = (-y_.T.dot(X_)).T

        A = np.eye(dim)
        # Don't restrict intercept to be positive
        if self.fit_intercept:
            A[-1, -1] = 0
        A = sparse.csc_matrix(A)

        l_matrix = np.zeros(dim)

        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=A, l=l_matrix, eps_abs=1e-8, eps_rel=1e-8, verbose=False)
        solution = solver.solve()
        weights = solution.x

        if self.fit_intercept:
            self.coef_ = weights[0:-1]
            self.intercept_ = weights[-1]
        else:
            self.coef_ = weights

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_ = X.values.copy()

        n_samples = X_.shape[0]

        # Augment features for intercept
        if self.fit_intercept:
            X_ = np.hstack([X_, np.ones(n_samples).reshape(-1, 1)])
            weights = np.concatenate([self.coef_, [self.intercept_]])
        else:
            weights = self.coef_

        y_pred = X_.dot(weights)
        return y_pred
