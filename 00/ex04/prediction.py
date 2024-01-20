import numpy as np
from tools import add_intercept

def predict_(x, theta):
	if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray) or
	 len(x.shape) != 2 or x.shape[1] != 1 or theta.shape != (2, 1)):
		return None
	return add_intercept(x) @ theta