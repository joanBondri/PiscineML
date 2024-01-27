import numpy as np

def simple_predict(x, theta):
	if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray) or
	 len(x.shape) != 2 or x.shape[1] != 1 or theta.shape != (2, 1)):
		return None
	resList = [theta[0] + theta[1] * xi for xi in x]
	return np.array(resList)