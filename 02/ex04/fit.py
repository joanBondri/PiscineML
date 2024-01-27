import numpy as np
from gradient import gradient
from tools import add_intercept

def fit_(x, y, theta, alpha, max_iter):
	if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray) or not isinstance(y, np.ndarray) or
	 len(x.shape) != 2 or len(y.shape) != 2 or
	 len(y) == 0 or y.shape[0] != x.shape[0] or y.shape[1] != 1 or
	 theta.shape[0] != x.shape[1] + 1 or theta.shape[1] != 1):
		return None
	if not isinstance(alpha, float) or not isinstance(max_iter, int):
		return None
	new_theta = theta.astype(np.float64)
	for i in range(max_iter):
		res = gradient(x, y, new_theta)
		if (res is None):
			return None
		new_theta -= alpha * res
	return new_theta