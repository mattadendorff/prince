"""Correspondence Analysis (CA)"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import base
from sklearn import utils

from . import util
from . import svd


class CA(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, n_components=2, n_iter=10, copy=True, random_state=None, engine='auto'):
        self.n_components = n_components
        self.n_iter = n_iter
        self.copy = copy
        self.random_state = random_state
        self.engine = engine

    def fit(self, X, y=None):

        # Check input
        utils.check_array(X)

        # Check all values are positive
        if np.any(X < 0):
            raise ValueError("All values in X should be positive")

        _, row_names, _, col_names = util.make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.copy:
            X = np.copy(X)

        # Compute the correspondence matrix which contains the relative frequencies
        X = X / np.sum(X)

        # Compute row and column masses
        self.row_masses_ = pd.Series(X.sum(axis=1), index=row_names)
        self.col_masses_ = pd.Series(X.sum(axis=0), index=col_names)

        # Compute standardised residuals
        r = self.row_masses_.values
        c = self.col_masses_.values
        S = sparse.diags(r ** -0.5) @ (X - np.outer(r, c)) @ sparse.diags(c ** -0.5)

        # Compute SVD on the standardised residuals
        self.U_, self.s_, self.V_ = svd.compute_svd(
            X=S,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine
        )

        # Compute total inertia
        self.total_inertia_ = (S @ S.T).trace()

        return self

    def transform(self, X):
        """Computes the row principal coordinates of a dataset.

        Same as calling `row_coordinates`. In most cases you should be using the same
        dataset as you did when calling the `fit` method. You might however also want to included
        supplementary data.
        """
        utils.validation.check_is_fitted(self, 's_')
        utils.check_array(X)
        return self.row_coordinates(X)

    @property
    def eigenvalues_(self):
        """The eigenvalues associated with each principal component."""
        utils.validation.check_is_fitted(self, 's_')
        return np.square(self.s_).tolist()

    @property
    def explained_inertia_(self):
        """The percentage of explained inertia per principal component."""
        utils.validation.check_is_fitted(self, 'total_inertia_')
        return [eig / self.total_inertia_ for eig in self.eigenvalues_]

    def row_coordinates(self, X):
        """The row principal coordinates."""
        utils.validation.check_is_fitted(self, 'V_')

        _, row_names, _, _ = util.make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.copy:
            X = np.copy(X)

        # Make sure the rows sum up to 1
        X = X / X.sum(axis=1)[:, None]

        return pd.DataFrame(
            data=X @ sparse.diags(self.col_masses_ ** -0.5) @ self.V_.T,
            index=row_names
        )

    def column_coordinates(self, X):
        """The column principal coordinates."""
        utils.validation.check_is_fitted(self, 'V_')

        _, _, _, col_names = util.make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.copy:
            X = np.copy(X)

        # Transpose and make sure the rows sum up to 1
        X = X.T / X.T.sum(axis=1)[:, None]

        return pd.DataFrame(
            data=X @ sparse.diags(self.row_masses_ ** -0.5) @ self.U_,
            index=col_names
        )
