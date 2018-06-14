import logging
from pprint import pprint
import numpy as np
import pandas as pd


class StandardScalerWithNaN(object):
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.columns_ = None

    def fit(self, X: pd.DataFrame):
        self.mean_ = []
        self.std_ = []
        self.columns_ = X.columns
        for i, column in enumerate(self.columns_):
            logging.info("Standard Scaler fitting column {}".format(column))
            # Get the column where it is not nan
            x_col = X[column].values
            x_col = x_col[~np.isnan(x_col)]
            self.mean_.append(np.mean(x_col))
            self.std_.append(np.std(x_col))
            logging.info("Fit column {} with mean {} and std {}".format(
                column, self.mean_[i], self.std_[i]))

    def transform(self, X: pd.DataFrame):
        assert np.all(self.columns_ == X.columns)
        X_out = X.astype(np.float32)
        for i, column in enumerate(self.columns_):
            logging.info("Transforming column {}".format(column))
            x_col = X_out[column].values
            nan_mask = np.isnan(x_col)
            x_col_no_nan = x_col[~nan_mask]
            # Standardize
            x_col_no_nan -= self.mean_[i]
            x_col_no_nan /= self.std_[i]
            x_col[~nan_mask] = x_col_no_nan
            x_col[nan_mask] = 0.
            # Replace the values
            X_out.values[:, i] = x_col
        return X_out


class CategoricalEncoderWithNaN(object):
    def __init__(self,
                 handle_unknown='ignore',
                 nan_values=("nan", "NaN", "XNA", "xna", "None", "none"),
                 reduce_binary_categories=False):
        self.categories_ = None
        self.n_values_ = None
        self.columns_ = None
        self.handle_unknown = handle_unknown
        self.nan_values = np.array(nan_values)
        self.reduce_binary_categories = reduce_binary_categories

    def __len__(self):
        return len(self.categories_)

    def fit_col(self, x_col):
        assert x_col.ndim == 1
        # Figure out what kind of nan values there are (for logging)
        is_nan_of_type = x_col[:, None] == self.nan_values[None, :]
        has_nan = np.any(is_nan_of_type, axis=0)
        logging.info("Column has nan values: {}".format(
            self.nan_values[has_nan]))
        # Figure out how many categories there are
        is_nan = np.any(is_nan_of_type, axis=1)
        categories = np.unique(x_col[~is_nan])
        counts = {
            category: np.count_nonzero(x_col == category)
            for category in categories
        }
        # Sort the categories based on frequency
        categories = np.array(
            sorted(
                categories,
                key=lambda category: counts[category],
                reverse=True))
        n_values = np.array([counts[category] for category in categories])
        logging.info("Discovered categories and counts:")
        pprint(list(zip(categories, n_values)))
        return categories, n_values

    def fit(self, X):
        self.categories_ = []
        self.n_values_ = []
        self.columns_ = X.columns
        for column in self.columns_:
            logging.info("Categorical fitting column {}".format(column))
            categories, n_values = self.fit_col(X[column].values)
            self.categories_.append(categories)
            self.n_values_.append(n_values)
        return

    def transform_col(self, x_col, categories):
        logging.info("Transforming into {} categories".format(len(categories)))
        x_out = (x_col[:, None] == categories[None, :])
        if self.handle_unknown == "error" and np.any(np.all(~x_out, axis=1)):
            raise ValueError("Unknown categories found")
        # Reduce it if just 2 categories
        if x_out.shape[1] == 2 and self.reduce_binary_categories:
            x_out = x_out[:, 1:2]
        return x_out.astype(np.float32)

    def transform(self, X):
        assert np.all(self.columns_ == X.columns)
        X_out = pd.DataFrame()
        for i, column in enumerate(self.columns_):
            logging.info("Transforming column {}".format(column))
            categories = self.categories_[i]
            # Pop and transform the column
            x_col = X[column].values
            x_out = self.transform_col(x_col, categories)
            # Deal with binary categories
            if len(categories) == 2 and self.reduce_binary_categories:
                assert x_out.shape[1] == 1
                logging.info("Column {} is a binary column".format(column))
                X_out[column + "=" + categories[1]] = x_out[:, 1]
                continue

            assert x_out.shape[1] == len(categories)
            # Add in the new one hot columns
            for i in range(len(categories)):
                X_out[column + "=" + categories[i]] = x_out[:, i]
        return X_out.astype(np.float32)
