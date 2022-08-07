import joblib
import numpy as np
from xgboost import XGBRegressor


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)

        