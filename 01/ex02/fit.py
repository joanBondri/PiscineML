import numpy as np
from pred import predict_
from tools import is_column_vector, add_intercept, transform_row_vector_to_column_vector
from cost_function import cost_

def calcule_derivative(theta, x, y):
	numberpred = len(x)
	X = add_intercept(x)
	res = (X.T @ (X @ theta - y)) / numberpred
	return res

def fit_(x, y, theta, alpha, max_iter):
	if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray) or
	 len(x.shape) != 2 or x.shape[1] != 1 or y.shape != x.shape or theta.shape != (2, 1)):
		return None
	if not isinstance(alpha, float) or not isinstance(max_iter, int):
		return None
	new_theta = theta.astype(np.float64)
	for i in range(max_iter):
		res = calcule_derivative(new_theta, x, y)
		if (res is None):
			return None
		new_theta -= alpha * res
	return new_theta