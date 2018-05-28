"""Multiple Correspondence Analysis (MCA)"""
import numpy as np
import pandas as pd
from sklearn import utils

from . import ca
from . import one_hot


class MCA(ca.CA):

    def fit(self, X, y=None):

        utils.check_array(X, dtype=[str, np.number])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_initial_columns = X.shape[1]

        # One-hot encode the data
        self.one_hot_ = one_hot.OneHotEncoder().fit(X)

        # Apply CA to the indicator matrix
        super().fit(self.one_hot_.transform(X))

        # Compute the total inertia
        n_new_columns = len(self.one_hot_.column_names_)
        self.total_inertia_ = (n_new_columns - n_initial_columns) / n_initial_columns

        return self

    def row_coordinates(self, X):
        return super().row_coordinates(self.one_hot_.transform(X))

    def column_coordinates(self, X):
        return super().column_coordinates(self.one_hot_.transform(X))

    def transform(self, X):
        """Computes the row principal coordinates of a dataset."""
        utils.validation.check_is_fitted(self, 's_')
        utils.check_array(X, dtype=[str, np.number])
        return self.row_coordinates(X)
