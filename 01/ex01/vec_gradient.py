import numpy as np
from tools import add_intercept

def simple_gradient(x, y, theta):
	if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray) or
	 len(x.shape) != 2 or x.shape[1] != 1 or y.shape != x.shape or theta.shape != (2, 1)):
		return None
	numberpred = len(x)
	X = add_intercept(x)
	res = (X.T @ (X @ theta - y)) / numberpred
	return res