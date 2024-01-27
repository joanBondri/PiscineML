import numpy as np

def add_intercept(arr):
	x = arr
	if (not isinstance(x, np.ndarray) or len(x.shape) != 2 or len(x[0]) == 0):
		return None
	ones = np.full((x.shape[0], 1), 1)
	return np.concatenate((ones, x), axis=1)