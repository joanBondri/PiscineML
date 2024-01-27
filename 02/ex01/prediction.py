import numpy as np
from tools import add_intercept

def predict_(x, theta):
	if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray) or
	 len(x.shape) != 2 or theta.shape[0] != x.shape[1] + 1 or theta.shape[1] != 1):
		return None
	X = x.astype(float)
	return add_intercept(X) @ theta