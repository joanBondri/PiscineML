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

def plot(x, y, theta, b_legend = True,axes_labels = ["x", "y"], data_labels = {"raw":"raw", "prediction":"prediction"}):
	try:
		fig, axes = plt.subplots(1,1, figsize=(10,8))
		axes.scatter(x, y, label = data_labels['raw'], c='#101214')
		axes.plot(x, predict_(x, theta), label = data_labels['prediction'], c='#4287f5')
		plt.legend()
		plt.xlabel(axes_labels[0])
		plt.ylabel(axes_labels[1])
		if b_legend:
			plt.legend()
		plt.grid()
		plt.show()
	except:
		return None