import numpy as np

def simple_gradient(x, y, theta):
	if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray) or
	 len(x.shape) != 2 or x.shape[1] != 1 or y.shape != x.shape or theta.shape != (2, 1)):
		return None
	numberpred = len(x)
	allDeriv = np.zeros((numberpred, 2))
	for row in range(allDeriv.shape[0]):
		allDeriv[row][0] = theta[0] + x[row][0] * theta[1] - y[row][0]
		allDeriv[row][1] = (theta[0] + x[row][0] * theta[1] - y[row][0]) * x[row][0]
	newTheta = np.sum(allDeriv, axis=0)
	newTheta = newTheta / numberpred
	return newTheta.reshape(-1, 1)
