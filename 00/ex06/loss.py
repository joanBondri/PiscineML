import numpy as np
from tools import is_vector, transform_line_vector_to_column_vector

def loss_elem_(y, y_hat):
	if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or
	 y.shape != y_hat.shape or len(y.shape) != 2 or y.shape[1] != 1):
		return None
	fy_hat = y_hat.astype(float)
	fy = y.astype(float)
	return (fy_hat - fy) ** 2

def loss_(y, y_hat):
	loss = loss_elem_(y, y_hat)
	if (loss is None):
		return
	return loss.sum() / (2 * len(y))