from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import numpy as np


class Preprocessor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        # keep column with int type
        X_filtered = X.select_dtypes(include=["int", "float"])
        # drop columns with nan values
        X = X_filtered.dropna(axis=1)
        # TODO: Improve drop and
        # TODO: Impute missing values
        # TODO: Encode categorical variables
        return X


class Classifier(BaseEstimator):
    def __init__(self):
        self.model1 = LogisticRegression(max_iter=10_000)
        self.model2 = LogisticRegression(max_iter=10_000)

    def fit(self, X, Y):
        # Y are pd.DataFrame here
        self.model1.fit(X, Y.iloc[:, 0])
        self.model2.fit(X, Y.iloc[:, 1])

    def predict(self, X):
        y1 = self.model1.predict_proba(X)
        y2 = self.model2.predict_proba(X)
        # Y_pred are nd.ndarray here
        Y_pred = np.concatenate([y1, y2], axis=1)
        # 2 discrete probability distributions
        assert Y_pred.shape[1] == 14
        return Y_pred


def get_estimator():
    pipe = make_pipeline(Preprocessor(), StandardScaler(), Classifier())
    return pipe
