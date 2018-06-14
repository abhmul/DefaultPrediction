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

    def transform_application_data(self, app_data):

        continuous_names = list(APPLICATION_CONT)
        discrete_names = list(APPLICATION_DIS)
        cat_names = list(APPLICATION_CAT)
        logging.info("Adding new features")
        # Amount loaned relative to salary
        app_data['LOAN_INCOME_RATIO'] = app_data['AMT_CREDIT'] / app_data[
            'AMT_INCOME_TOTAL']
        app_data['ANNUITY_INCOME_RATIO'] = app_data['AMT_ANNUITY'] / app_data[
            'AMT_INCOME_TOTAL']
        continuous_names.extend(['LOAN_INCOME_RATIO', 'ANNUITY_INCOME_RATIO'])

        # Number of overall payments (I think!)
        app_data['ANNUITY LENGTH'] = app_data['AMT_CREDIT'] / app_data[
            'AMT_ANNUITY']
        continuous_names.extend(['ANNUITY LENGTH'])

        # Social features
        app_data['WORKING_LIFE_RATIO'] = app_data['DAYS_EMPLOYED'] / app_data[
            'DAYS_BIRTH']
        app_data['INCOME_PER_FAM'] = app_data['AMT_INCOME_TOTAL'] / app_data[
            'CNT_FAM_MEMBERS']
        app_data['CHILDREN_RATIO'] = app_data['CNT_CHILDREN'] / app_data[
            'CNT_FAM_MEMBERS']
        continuous_names.extend(
            ['WORKING_LIFE_RATIO', 'INCOME_PER_FAM', 'CHILDREN_RATIO'])

        # Apply standard scalar to all continuous columns
        logging.info("Transforming continuous features for app_data to "
                     "Standard normal")
        cont_app_data = app_data[continuous_names]
        if self.application_scaler is None:
            logging.info("No Scaler, fitting one.")
            self.application_scaler = StandardScalerWithNaN()
            self.application_scaler.fit(cont_app_data)
        app_data[continuous_names] = self.application_scaler.transform(
            cont_app_data)

        # Turn the categorical data into one-hot
        logging.info("Transforming categorical features for app_data to "
                     "OneHot")
        cat_app_data = app_data[cat_names].fillna("nan")
        if self.application_encoder is None:
            logging.info("No encoder, fitting one.")
            self.application_encoder = CategoricalEncoderWithNaN()
            self.application_encoder.fit(cat_app_data)
        cat_features = self.application_encoder.transform(cat_app_data)
        # Drop the categorical features and add the one-hot features
        app_data.drop(cat_names, axis=1, inplace=True)
        app_data[cat_features.columns] = cat_features

        # Fill the discrete nan values with 0
        logging.info("Transforming nan discreate features for app_data "
                     "to 0")
        app_data[discrete_names] = app_data[discrete_names].fillna(0)

        return app_data.astype(np.float32)

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
