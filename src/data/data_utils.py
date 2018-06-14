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

from .data_globals import APPLICATION_CAT, APPLICATION_CONT, APPLICATION_DIS
from .transformers import StandardScalerWithNaN, CategoricalEncoderWithNaN


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

        self.application_encoder = None
        self.application_scaler = None

        # Load the application data to fit the encoder and scaler
        self.load_application_data(type='both')

    def transform_application_data(self, application_df):

        # Apply standard scalar to all continuous columns
        logging.info("Transforming continuous features for application_df to "
                     "Standard normal")
        cont_application_df = application_df[APPLICATION_CONT]
        if self.application_scaler is None:
            logging.info("No Scaler, fitting one.")
            self.application_scaler = StandardScalerWithNaN()
            self.application_scaler.fit(cont_application_df)
        application_df[APPLICATION_CONT] = self.application_scaler.transform(
            cont_application_df)

        # Turn the categorical data into one-hot
        logging.info("Transforming categorical features for application_df to "
                     "OneHot")
        cat_application_df = application_df[APPLICATION_CAT].fillna("nan")
        if self.application_encoder is None:
            logging.info("No encoder, fitting one.")
            self.application_encoder = CategoricalEncoderWithNaN()
            self.application_encoder.fit(cat_application_df)
        cat_features = self.application_encoder.transform(cat_application_df)
        # Drop the categorical features and add the one-hot features
        application_df.drop(APPLICATION_CAT, axis=1, inplace=True)
        application_df[cat_features.columns] = cat_features

        # Fill the discrete nan values with 0
        logging.info("Transforming nan discreate features for application_df "
                     "to 0")
        application_df[APPLICATION_DIS] = application_df[
            APPLICATION_DIS].fillna(0)
        return application_df.astype(np.float32)

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
        logging.info("Application Data:\n{}".format(data.tail()))
        if type == 'train':
            return ids, data, targets
        return ids, data

    def load_train(self):
        # TODO: For now just loads the trianing data
        ids, data, targets = self.load_application_data(type='train')
        y = targets.values.astype(np.float32)[:, None]
        return NpDataset(data.values, y=y, ids=ids.values)

    def load_test(self):
        ids, data = self.load_application_data(type='test')
        return NpDataset(data.values, ids=ids.values)

    @staticmethod
    def save_submission(submission_path, predictions, ids):
        logging.info("Saving submission to {}".format(submission_path))
        out_df = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': predictions})
        out_df.to_csv(submission_path, index=False)
