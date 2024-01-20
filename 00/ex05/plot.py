import numpy as np
import matplotlib.pyplot as plt

def add_intercept(arr : np.ndarray):
	x = arr
	lenX = len(x)
	if (x.ndim > 2):
		return
	if (x.ndim == 1):
		x = x.reshape(-1, 1)
	ones = np.full((x.shape[0], 1), 1)
	return np.concatenate((ones, x), axis=1)

def predict_(x, theta):
	if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray) or
	theta.shape != (2, 1)):
		return None
	return np.sum(add_intercept(x) * theta.reshape(1, -1), axis=1)

def plot(x, y, theta):
	plt.scatter(x, y, color="green")
	yhat = predict_(x, theta)
	plt.plot(x, yhat, color="red")
	plt.show()