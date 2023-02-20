import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows.sklearn_pipeline import SKLearnPipeline, Estimator
from rampwf.prediction_types import make_multiclass
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, f1_score
from pathlib import Path
import requests
import json
import io
import os

problem_title = "TODO"

# Model
workflow = Estimator()


# Cross-validate
def get_cv(X, y):
    cv = KFold(n_splits=5)
    for train_index, test_index in cv.split(X, y):
        yield train_index, test_index


# Scores
score_types = [
    rw.score_types.RMSE(name="rmse", precision=3),
]

# Predictions
Predictions = rw.prediction_types.make_regression()


# Data
def _read_data(path, type_):
    data_dir = Path(path) / "data"

    if type_ == "train":
        X_train = pd.read_parquet(data_dir / "data_train.parquet")
        y_train = pd.read_csv(data_dir / "labels_train.csv", index_col=0)
        return X_train, y_train
    elif type_ == "test":
        X_test = pd.read_parquet(data_dir / "data_test.parquet")
        y_test = pd.read_csv(data_dir / "labels_test.csv", index_col=0)
        return X_test, y_test
    else:
        raise Exception("type_ must be 'train' or 'test'")


def get_train_data(path="."):
    return _read_data(path, "train")


def get_test_data(path="."):
    return _read_data(path, "test")
