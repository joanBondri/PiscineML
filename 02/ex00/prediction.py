import numpy as np
from tools import add_intercept

def simple_predict(x, theta):
	if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray) or
	 len(x.shape) != 2 or theta.shape[0] != x.shape[1] + 1 or theta.shape[1] != 1):
		return None
	m = x.shape[0]
	n = x.shape[1] + 1
	X = add_intercept(x)
	res = np.zeros((m, 1))
	for row in range(m):
		for col in range(n):
			res[row][0] += X[row][col] * theta[col][0]
	return res