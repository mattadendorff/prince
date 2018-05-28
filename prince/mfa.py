"""Multiple Factor Analysis (MFA)"""
import itertools
import numpy as np
import pandas as pd
from sklearn import utils

from . import mca
from . import pca


class MFA(pca.PCA):

    def __init__(self, groups=None, rescale_with_mean=True, rescale_with_std=True, n_components=2,
                 n_iter=10, copy=True, random_state=None, engine='auto'):
        super().__init__(
            rescale_with_mean=rescale_with_mean,
            rescale_with_std=rescale_with_std,
            n_components=n_components,
            n_iter=n_iter,
            copy=copy,
            random_state=random_state,
            engine=engine
        )
        self.groups = groups

    def fit(self, X, y=None):

        # Checks groups are provided
        if self.groups is None:
            raise ValueError('Groups have to be specified')

        # Check input
        utils.check_array(X, dtype=[str, np.number])

        # Make sure X is a DataFrame for convenience
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Check group types are consistent
        self.all_nums_ = {}
        for name, cols in sorted(self.groups.items()):
            all_num = all(pd.api.types.is_numeric_dtype(X[c]) for c in cols)
            all_cat = all(pd.api.types.is_string_dtype(X[c]) for c in cols)
            if not (all_num or all_cat):
                raise ValueError('Not all columns in "{}" group are of the same type'.format(name))
            self.all_nums_[name] = all_num

        # Run a factor analysis in each group
        self.partial_factor_analysis_ = {}
        for name, cols in sorted(self.groups.items()):
            if self.all_nums_[name]:
                fa = pca.PCA(
                    rescale_with_mean=self.rescale_with_mean,
                    rescale_with_std=self.rescale_with_std,
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=self.copy,
                    random_state=self.random_state,
                    engine=self.engine
                )
            else:
                fa = mca.MCA(
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    copy=self.copy,
                    random_state=self.random_state,
                    engine=self.engine
                )
            self.partial_factor_analysis_[name] = fa.fit(X[cols])

        # Fit the global PCA
        super().fit(self._build_X_global(X))

        return self

    def _build_X_global(self, X):

        # Make sure X is a DataFrame for convenience
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return pd.concat(
            (
                X[cols] / self.partial_factor_analysis_[name].s_[0]
                if self.all_nums_[name]
                else X[cols]
                for name, cols in sorted(self.groups.items())
            ),
            axis='columns'
        )

    def transform(self, X):
        """Returns the row principal coordinates of a dataset."""
        utils.validation.check_is_fitted(self, 's_')
        utils.check_array(X)
        return self.row_coordinates(X)

    def row_coordinates(self, X):
        """Returns the row principal coordinates."""
        utils.validation.check_is_fitted(self, 's_')
        n = X.shape[0]
        return n ** 0.5 * super().row_coordinates(self._build_X_global(X))

    def row_contributions(self, X):
        """Returns the row contributions towards each principal component."""
        utils.validation.check_is_fitted(self, 's_')
        return super().row_contributions(self._build_X_global(X))

    def partial_row_coordinates(self, X):
        """Returns the row coordinates for each group."""
        utils.validation.check_is_fitted(self, 's_')

        # Make sure X is a DataFrame for convenience
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Define the projection matrix P
        n = X.shape[0]
        P = n ** 0.5 * self.U_ / self.s_

        # Get the projections for each group
        coords = {}
        for name, cols in sorted(self.groups.items()):
            X_partial = X[cols].values
            Z_partial = X_partial / self.partial_factor_analysis_[name].s_[0]
            coords[name] = len(self.groups) * (Z_partial @ Z_partial.T) @ P

        # Convert coords to a MultiIndex DataFrame
        coords = pd.DataFrame({
            (name, i): group_coords[:, i]
            for name, group_coords in coords.items()
            for i in range(group_coords.shape[1])
        })

        return coords

