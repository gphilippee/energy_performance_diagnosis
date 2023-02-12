from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def get_estimator():
    pipe = make_pipeline(LinearRegression())
    return pipe
