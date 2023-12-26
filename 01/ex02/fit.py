import numpy as np
from pred import predict_
from tools import is_column_vector, add_intercept, transform_row_vector_to_column_vector
from cost_function import cost_

def calcule_derivative(theta, x, y):
	prediction = predict_(theta, x)
	x_intercept = add_intercept(x)
	if prediction is None or x_intercept is None or not is_column_vector(y) or x.shape[0] != y.shape[0] :
		return
	diff = prediction - y
	derivative_sum = np.sum((diff * x_intercept) / len(y), axis=0)
	return transform_row_vector_to_column_vector(derivative_sum)

def fit_(theta, x, y, alpha=.1, n_cycle=10000):
	new_theta = theta
	for i in range(n_cycle):
		res = calcule_derivative(new_theta, x, y)
		if (res is None):
			return
		new_theta -= alpha * res
	return new_theta