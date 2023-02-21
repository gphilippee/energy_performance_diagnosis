import warnings

import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows.sklearn_pipeline import SKLearnPipeline, Estimator
from rampwf.prediction_types import make_multiclass
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.prediction_types.base import BasePrediction

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


class CELogLoss(BaseScoreType):
    # subclass BaseScoreType to use raw y_pred (proba's)
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name="ce_ll", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        score = log_loss(y_true[:, 1:], y_pred[:, 1:])
        return score


class GESLogLoss(BaseScoreType):
    # subclass BaseScoreType to use raw y_pred (proba's)
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name="ges_ll", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        score = log_loss(y_true[:, 1:], y_pred[:, 1:])
        return score


class CEF1Score(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="ce_f1", precision=2, average="macro"):
        self.name = name
        self.precision = precision
        self.average = average

    def __call__(self, y_true_label_index, y_pred_label_index):
        return f1_score(
            y_true_label_index[0], y_pred_label_index[0], average=self.average
        )


class GESF1Score(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="ges_f1", precision=2, average="macro"):
        self.name = name
        self.precision = precision
        self.average = average

    def __call__(self, y_true_label_index, y_pred_label_index):
        return f1_score(
            y_true_label_index[1], y_pred_label_index[1], average=self.average
        )


# Scores
score_types = [
    GESLogLoss(name="ges_ll", precision=2),
    CELogLoss(name="ce_ll", precision=2),
    GESF1Score(name="ges_f1", precision=2, average="macro"),
    CEF1Score(name="ce_f1", precision=2, average="macro"),
]

# Predictions
BaseMultiClassPredictions = rw.prediction_types.make_multiclass(
    label_names=["A", "B", "C", "D", "E", "F", "G"]
)


class Predictions(BaseMultiClassPredictions):
    """
    Overriding parts of the ramp-workflow version to preserve the y_pred /
    y_true DataFrames.
    """

    n_columns = 14

    def __init__(self, y_pred=None, y_true=None, n_samples=None, fold_is=None):
        # override init to not convert y_pred/y_true to arrays
        if y_pred is not None:
            if fold_is is not None:
                y_pred = y_pred[fold_is]
            self.y_pred = np.array(y_pred)
        elif y_true is not None:
            if fold_is is not None:
                y_true = y_true.iloc[fold_is]
            self._init_from_pred_labels(y_true)
        elif n_samples is not None:
            self.y_pred = np.empty((n_samples, self.n_columns), dtype=float)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError("Missing init argument: y_pred, y_true, or n_samples")
        self.check_y_pred_dimensions()

    def _init_from_pred_labels(self, y_pred_labels):
        """
        Initialize y_pred from y_pred_labels.
        We set to 1.0 the columns corresponding to the labels in y_pred_labels.

        Example: label_names = ['A', 'B', 'C']
        y_pred_labels = pd.DataFrame([['A', 'B'], ['B', 'C'], ['C', 'A']])
        y_pred = np.array([[1, 0, 0, 0, 1, 0],
                            [0, 1, 0, 0, 0, 1],
                            [0, 0, 1, 1, 0, 0]])

        :param y_pred_labels:
        :return:
        """
        self.y_pred = np.zeros(
            (len(y_pred_labels), len(self.label_names) * 2), dtype=np.float64
        )
        for ps_i, label1, label2 in zip(
            self.y_pred, y_pred_labels.iloc[:, 0], y_pred_labels.iloc[:, 1]
        ):
            ps_i[self.label_names.index(label1)] = 1.0
            ps_i[self.label_names.index(label2) + 7] = 1.0

    @property
    def y_pred_label_index(self):
        """Multi-class y_pred is the index of the predicted label."""
        # y_pred are np.ndarray here
        return (
            np.argmax(self.y_pred[:, :7], axis=1),
            np.argmax(self.y_pred[:, 7:], axis=1),
        )

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        if index_list is None:  # we combine the full list
            index_list = range(len(predictions_list))
        y_comb_list = np.array([predictions_list[i].y_pred for i in index_list])
        # clipping probas into [0, 1], also taking care of the case of all
        # zeros
        y_comb_list[:, :, 1:] = np.clip(y_comb_list[:, :, 1:], 10**-15, 1 - 10**-15)
        # normalizing probabilities
        y_comb_list[:, :, 1:] = y_comb_list[:, :, 1:] / np.sum(
            y_comb_list[:, :, 1:], axis=2, keepdims=True
        )
        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            y_comb = np.nanmean(y_comb_list, axis=0)
        combined_predictions = cls(y_pred=y_comb)
        return combined_predictions


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
