import numpy as np
from tools import is_vector, transform_line_vector_to_column_vector

def loss_elem_(y : np.ndarray, y_hat : np.ndarray):
	yh = transform_line_vector_to_column_vector(y_hat)
	yp = transform_line_vector_to_column_vector(y)
	if yh is None or yp is None or yh.shape != yp.shape :
		return
	return (yh - yp) ** 2

def loss_(y : np.ndarray, y_hat : np.ndarray):
	loss = loss_elem_(y, y_hat)
	if (loss is None):
		return
	return loss.sum() / (2 * len(y))