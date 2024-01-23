import numpy as np
from tools import is_vector, transform_line_vector_to_column_vector

def loss_(y : np.ndarray, y_hat : np.ndarray):
	if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or
	 y.shape != y_hat.shape or len(y.shape) != 2 or y.shape[1] != 1):
		return None
	fy_hat = y_hat.astype(float)
	fy = y.astype(float)
	diff = fy_hat - fy
	vec_res = diff.T @ diff
	return 1 / (2 * len(y)) * vec_res[0][0]