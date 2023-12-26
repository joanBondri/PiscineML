import numpy as np

def add_intercept(arr : np.ndarray):
	x = arr
	lenX = len(x)
	if (x.ndim > 2):
		return
	if (x.ndim == 1):
		x = x.reshape(-1, 1)
	ones = np.full((x.shape[0], 1), 1)
	return np.concatenate((ones, x), axis=1)