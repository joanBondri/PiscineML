import numpy as np
from math import sqrt

def mse_(y, y_hat):
    try:
        fy_hat = y_hat.astype(float)
        fy = y.astype(float)
        diff = fy_hat - fy
        vec_res = diff.T @ diff
        return float(1 / (len(y)) * vec_res[0][0])
    except:
        return None


def rmse_(y, y_hat):
    try:
        return float(sqrt(mse_(y, y_hat)))
    except:
        return None


def mae_(y, y_hat):
    try:
        mae = (1.0 / y.shape[0]) * np.sum(np.absolute(y - y_hat), axis=0)
        return float(mae)
    except:
        return None



def r2score_(y, y_hat):
    try:
        mean = np.mean(y, axis = 0)
        r2 = 1 - np.sum((y_hat - y) ** 2, axis = 0) / np.sum((y - mean) ** 2, axis = 0)
        return float(r2)
    except:
        return None