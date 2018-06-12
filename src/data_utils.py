"""
For now just use application train and test for getting the workflow setup.
We'll add more stuff later
"""
import os
import glob
import logging
from pprint import pprint

import pandas as pd
import numpy as np
from pyjet.data import NpDataset
from sklearn.preprocessing import StandardScaler
from data_globals import APPLICATION_CAT, APPLICATION_CONT


# TODO: Make the standard scaler and encoder take the whole pandas df
# as an input instead
# TODO: Add a ZeroOneScaler (brings discrete features into 0-1 range, turn nans
# into 0)
class StandardScalerWithNaN(object):
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

        self.scalers = []

    def init_scaler(self):
        return StandardScaler(
            copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)

    def fit(self, X):
        self.scalers = []
        for i in range(X.shape[1]):
            x_col = X[:, i]
            x_col = x_col[~np.isnan(x_col), None]
            self.scalers.append(self.init_scaler())
            self.scalers[i].fit(x_col)
            logging.info("Fit row {} with mean {} and variance {}".format(
                i, self.scalers[i].mean_, self.scalers[i].var_))

    def transform(self, X):
        for i in range(X.shape[1]):
            x_col = X[:, i]
            nan_mask = np.isnan(x_col)
            x_col_no_nan = x_col[~nan_mask, None]
            x_col[~nan_mask] = self.scalers[i].transform(x_col_no_nan)[:, 0]
            x_col[nan_mask] = 0.
            X[:, i] = x_col
        return X


class CategoricalEncoder(object):
    def __init__(self, handle_unknown='error'):
        self.categories_ = None
        self.n_values_ = None
        self.handle_unknown = handle_unknown

    def __len__(self):
        return len(self.categories_)

    def fit(self, X):
        categories = np.unique(X[X != "nan"])
        counts = {
            category: np.count_nonzero(X == category)
            for category in categories
        }
        # Sort the categories based on frequency
        self.categories_ = np.array(
            sorted(
                categories,
                key=lambda category: counts[category],
                reverse=True))
        self.n_values_ = np.array(
            [counts[category] for category in self.categories_])

        logging.info("Discovered categories {}, with counts {}".format(
            self.categories_, self.n_values_))

    def transform(self, X):
        X_out = (X[:, None] == self.categories_[None, :])
        if self.handle_unknown == "error" and np.any(np.all(~X_out, axis=1)):
            raise ValueError("Unknown categories found")
        # Reduce it if just 2 categories
        if X_out.shape[1] == 2:
            X_out = X_out[:, 1:2]
        return X_out.astype(np.float32)


def file_basename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


class HomeCreditData(object):
    def __init__(self, path_to_input="../input"):
        self.path_to_input = path_to_input
        # Discover the files in the input folder
        self.file_paths = glob.glob(os.path.join(self.path_to_input, "*.csv"))

        logging.info("Discovered files:")
        pprint(self.file_paths)

        for fp in self.file_paths:
            setattr(self, file_basename(fp), fp)

        self.application_encoder = {}
        self.application_scaler = None

        # Load the application data to fit the encoder and scaler
        self.load_application_data(type='both')

    def transform_application_data(self, application_df):

        # Apply standard scalar to all continuous columns
        logging.info("Transforming continuous features to Standard normal")
        if self.application_scaler is None:
            logging.info("No Scaler, fitting one.")
            self.application_scaler = StandardScalerWithNaN(
                copy=False, with_mean=True, with_std=True)
            cont_application_df = application_df[APPLICATION_CONT]
            self.application_scaler.fit(
                cont_application_df.values.astype(float))
        application_df[APPLICATION_CONT] = self.application_scaler.transform(
                cont_application_df.values)

        # Turn the categorical data into one-hot
        for column in APPLICATION_CAT:
            logging.info("Transforming {} to onehot".format(column))
            # Pop the column and fill in missing values
            categorical = application_df.pop(column)
            categorical = categorical.fillna("nan")

            if column not in self.application_encoder:
                logging.info("No encoder, fitting one.")
                self.application_encoder[column] = CategoricalEncoder(
                    handle_unknown='ignore')
                self.application_encoder[column].fit(categorical.values)
                logging.info("Found {} categories for column {}".format(
                    len(self.application_encoder[column]), column))
            onehot = self.application_encoder[column].transform(
                categorical.values)
            # Add in the new columns
            if len(self.application_encoder[column]) == 2:
                category = self.application_encoder[column].categories_[1]
                application_df[column + "=" + category] = onehot[:, 0]
                continue

            for i, category in enumerate(
                    self.application_encoder[column].categories_):
                application_df[column + "=" + category] = onehot[:, i]

        # Fill the rest of the nan values with 0
        application_df = application_df.fillna(0).astype(np.float32)
        return application_df

    def load_application_data(self, type="train"):
        assert type in {"train", "test", "both"}
        logging.info("Loading application data for {}".format(type))
        if type == 'test':
            data = pd.read_csv(self.application_test)
        elif type == 'train':
            data = pd.read_csv(self.application_train)
        else:
            train_data = pd.read_csv(self.application_train)
            test_data = pd.read_csv(self.application_test)
            data = pd.concat([test_data, train_data])
            type = 'train'

        if type == 'train':
            targets = data.pop("TARGET")
        ids = data.pop("SK_ID_CURR")

        # TODO: For now keeping it simple for nn, but later we need to
        # generalize this for all models

        # Convert all categorical features
        logging.info("Transforming application data")
        data = self.transform_application_data(data)
        if type == 'train':
            return ids, data, targets
        return ids, data

    def load_train(self):
        # TODO: For now just loads the trianing data
        ids, data, targets = self.load_application_data(type='train')
        return NpDataset(data.values, y=targets.values, ids=ids.values)
