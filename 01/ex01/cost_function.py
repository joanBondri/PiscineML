import numpy as np
from tools import is_column_vector, add_intercept
from pred import predict_
def cost_elem_(theta, x, y):
	prediction = predict_(theta, x)
	if prediction is None or not is_column_vector(y) or x.shape[0] != y.shape[0] :
		return
	diff = prediction - y
	return diff ** 2 / (2 * y.shape[0])

def cost_( theta, x, y):
	cost_elem = cost_elem_(theta, x, y)
	if (cost_elem is None):
		return
	return cost_elem.sum()