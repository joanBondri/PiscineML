from tools import is_column_vector, is_row_vector, is_vector, add_intercept
import numpy as np

def predict_(theta, x):
	if not is_column_vector(theta) or not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] + 1 != theta.shape[0]:
		print("Incompatible dimension match between X and theta.")
		return
	interceptingX = add_intercept(x)
	return interceptingX @ theta