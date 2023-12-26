import numpy as np
from tools import add_intercept, is_vector, transform_column_vector_to_row_vector

def predict_(x : np.ndarray, theta : np.ndarray):
	theta_prime = transform_column_vector_to_row_vector(theta)
	if (not is_vector(x) or theta_prime is None or len(theta) != 2):
		raise TypeError("Error: wrong arg type")
	return np.sum(add_intercept(x) * theta_prime, axis=1)