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

def predict_(x : np.ndarray, theta : np.ndarray):
	if (not (x.ndim == 1 and x.dtype.kind in ('i', 'f')) or
	 	not (theta.ndim == 2 and theta.dtype.kind in ('i', 'f') and len(theta) == 2)):
		raise TypeError("Error: wrong arg type")
	return np.sum(add_intercept(x) * theta.reshape(1, -1), axis=1)

def plot(x, y, theta):
	plt.scatter(x, y, color="green")
	yhat = predict_(x, theta)
	plt.plot(x, yhat, color="red")
	plt.title("What ?")
	plt.xlabel("X-axis")
	plt.ylabel("Y-axis")
	plt.show()