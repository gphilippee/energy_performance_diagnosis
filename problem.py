import warnings

import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows.sklearn_pipeline import Estimator
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, f1_score, confusion_matrix
from pathlib import Path

problem_title = "Energy performance diagnosis"

# Model
workflow = Estimator()


# Cross-validate
def get_cv(X, y):
    cv = KFold(n_splits=5)
    for train_index, test_index in cv.split(X, y):
        yield train_index, test_index

class ECLoss(BaseScoreType):
    """    
    Some errors (e.g. predicting class "G" when it is class "A") might count
    for more in the final scores. The missclassification weights were designed 
    to penalize more mistakes on buildings with low energy efficiency.

    Bilinear Loss : https://arxiv.org/pdf/1704.06062.pdf
    """
    
    # subclass BaseScoreType to use raw y_pred (proba's)
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name="ec_ll", precision=2, alpha=0.99):
        self.name = name
        self.precision = precision
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        
        L_CE = log_loss(y_true[:, 1:], y_pred[:, 1:])

        W = np.array(
            [
                [0, 5, 7, 10, 10, 10, 10],
                [5, 0, 3, 8, 10, 10, 10],
                [7, 3, 0, 4, 10, 10, 10],
                [10, 8, 4, 0, 9, 10, 10],
                [10, 10, 10, 9, 0, 8, 10],
                [10, 10, 10, 10, 8, 0, 9],
                [10, 10, 10, 10, 10, 9, 0],
            ]
        )
        W = W / np.max(W)
        n_classes = len(W)

        y_pred = np.argmax(y_pred[:, 1:], axis=1)
        y_true = np.argmax(y_true[:, 1:], axis=1)

        conf_mat = confusion_matrix(
            y_true, y_pred, labels=np.arange(n_classes)
        )

        n = len(y_true)
        L_B = np.multiply(conf_mat, W).sum() / n

        score = (1 - self.alpha) * L_CE + self.alpha * L_B

        return score


class GHGLogLoss(BaseScoreType):
    # subclass BaseScoreType to use raw y_pred (proba's)
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name="ghg_ll", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        score = log_loss(y_true[:, 1:], y_pred[:, 1:])
        return score


class ECF1Score(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="ec_f1", precision=2, average="macro"):
        self.name = name
        self.precision = precision
        self.average = average

    def __call__(self, y_true_label_index, y_pred_label_index):
        return f1_score(
            y_true_label_index[0], y_pred_label_index[0], average=self.average
        )


class GHGF1Score(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="ghg_f1", precision=2, average="macro"):
        self.name = name
        self.precision = precision
        self.average = average

    def __call__(self, y_true_label_index, y_pred_label_index):
        return f1_score(
            y_true_label_index[1], y_pred_label_index[1], average=self.average
        )


class Mixed(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name="mixed", precision=2):
        self.name = name
        self.precision = precision
        self.ghg_ll = GHGLogLoss()
        self.ec_ll = ECLoss()
        self.ghg_f1 = GHGF1Score()
        self.ec_f1 = ECF1Score()

    def score_function(self, ground_truths, predictions):
        return self.__call__(ground_truths, predictions)

    def __call__(self, ground_truths, predictions):
        # Log-loss
        y_true = ground_truths.y_pred
        y_pred = predictions.y_pred
        ghg_ll_score = self.ghg_ll(y_true, y_pred)
        ec_ll_score = self.ec_ll(y_true, y_pred)

        # F1 score
        y_true_label_index = ground_truths.y_pred_label_index
        y_pred_label_index = predictions.y_pred_label_index
        ghg_f1_score = self.ghg_f1(y_true_label_index, y_pred_label_index)
        ec_f1_score = self.ec_f1(y_true_label_index, y_pred_label_index)
        score = 0.5 * (ghg_ll_score + ec_ll_score) + 0.1 * (
            2 - ghg_f1_score - ec_f1_score
        )
        return score


# Scores
score_types = [
    Mixed(name="mixed", precision=2),
    GHGLogLoss(name="ghg_ll", precision=2),
    ECLoss(name="ec_ll", precision=2),
    GHGF1Score(name="ghg_f1", precision=2, average="macro"),
    ECF1Score(name="ec_f1", precision=2, average="macro"),
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
