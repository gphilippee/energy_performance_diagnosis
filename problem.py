import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows.sklearn_pipeline import SKLearnPipeline, Estimator
from rampwf.prediction_types import make_regression

from sklearn.model_selection import KFold

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
def get_train_data(path="."):
    """

    :param path:
    :return: X, y
    """
    return np.random.randn(100, 10), np.random.randn(100)


def get_test_data(path="."):
    """

    :param path:
    :return: X, y
    """
    return np.random.randn(100, 10), np.random.randn(100)
