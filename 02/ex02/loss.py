import numpy as np

def loss_(y : np.ndarray, y_hat : np.ndarray):
	if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or
	 y.shape != y_hat.shape or len(y.shape) != 2 or y.shape[1] != 1 or len(y) == 0):
		return None
	fy_hat = y_hat.astype(float)
	fy = y.astype(float)

	diff = fy_hat - fy
	dot = (diff.T @ diff)[0][0]
	return 1 / (2 * len(y)) * dot