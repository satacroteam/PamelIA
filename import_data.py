"""
Import data

TRAIN :

    count       8581
    unique         2
    top       benign
    freq        7902

    Concentration de benin dans le train : 92.0871693275842%

TEST :

    count       4291
    unique         2
    top       benign
    freq        3951

    Concentration de benin dans le test : 92.076439058494519%
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split


class ImportData(object):
    """
    Class to import data
    """
    def __init__(self):
        self.data_path = 'Donnees_apprentissage'
        self.test_path = 'test'

    def make_mini_train_valid_dataset(self):
        X = [path for path in os.listdir('Donnees_apprentissage')]
        y = pd.read_csv('label_learn.csv', sep=';')["label"].tolist()
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=500, test_size=300, stratify=y)
        # Fill the train dataset
        for picture, answer in zip(x_train, y_train):
            if answer == "benign":
                os.rename(
                os.path.join(self.data_path, picture),
                os.path.join("data", "train", "benign", picture))
            else:
                os.rename(
                os.path.join(self.data_path, picture),
                os.path.join("data", "train", "malignant", picture))

        # Fill the valid dataset
        for picture, answer in zip(x_test, y_test):
            if answer == "benign":
                os.rename(
                os.path.join(self.data_path, picture),
                os.path.join("data", "valid", "benign", picture))
            else:
                os.rename(
                os.path.join(self.data_path, picture),
                os.path.join("data", "valid", "malignant", picture))



