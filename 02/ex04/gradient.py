import numpy as np
from tools import add_intercept

def gradient(x, y, theta):
	if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray) or not isinstance(y, np.ndarray) or
	 len(x.shape) != 2 or len(y.shape) != 2 or
	 len(y) == 0 or y.shape[0] != x.shape[0] or y.shape[1] != 1 or
	 theta.shape[0] != x.shape[1] + 1 or theta.shape[1] != 1):
		return None
	X = x.astype(float)
	Y = y.astype(float)
	numberpred = len(X)
	Xp = add_intercept(X)
	res = (Xp.T @ (Xp @ theta - Y)) / numberpred
	return res