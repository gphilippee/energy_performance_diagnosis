import numpy as np
import pandas as pd
import rampwf as rw
from rampwf.workflows.sklearn_pipeline import SKLearnPipeline, Estimator
from rampwf.prediction_types import make_regression

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
def get_data():
    """

    :param path:
    :return: df
    """
    # URL de l'API pour les données DPE tertiaire
    url = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe-tertiaire/full"

    # Envoi de la requête
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.text))
    return df
