import numpy as np
from math import sqrt

def mse_(y, y_hat):
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or
	 y.shape != y_hat.shape or len(y.shape) != 2 or y.shape[1] != 1):
		return None
	fy_hat = y_hat.astype(float)
	fy = y.astype(float)
	diff = fy_hat - fy
	vec_res = diff.T @ diff
	return float(1 / (len(y)) * vec_res[0][0])

def rmse_(y, y_hat):
    return float(sqrt(mse_(y, y_hat)))

def mae_(y, y_hat):
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or
    y.shape != y_hat.shape or len(y.shape) != 2 or y.shape[1] != 1):
        return None
    mae = (1.0 / y.shape[0]) * np.sum(np.absolute(y - y_hat), axis=0)
    return float(mae)


def r2score_(y, y_hat):
    if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or
    y.shape != y_hat.shape or len(y.shape) != 2 or y.shape[1] != 1):
        return None
    mean = np.mean(y, axis = 0)
    r2 = 1 - np.sum((y_hat - y) ** 2, axis = 0) / np.sum((y - mean) ** 2, axis = 0)
    return float(r2)
