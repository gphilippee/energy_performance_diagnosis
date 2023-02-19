import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows.sklearn_pipeline import SKLearnPipeline, Estimator
from rampwf.prediction_types import make_regression
from sklearn.model_selection import train_test_split


from sklearn.model_selection import KFold
import requests
import json
import io
import pandas as pd

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
# Data
def _read_data(path, type_):

    fname = "data.csv"
    fp = os.path.join(path, "data", fname)
    data = pd.read_csv(fp, index_col=0)
    
    y = data[['classe_consommation_energie', 'estimation_ges']]
    X = data.drop(columns = ['classe_consommation_energie', 'estimation_ges'], axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3)
    
    if type_ == "train":
        return X_train, y_train
    
    else:
        return X_test, y_test
    

def get_train_data(path="."):
    return _read_data(path, "train")


def get_test_data(path="."):
    return _read_data(path, "test")
