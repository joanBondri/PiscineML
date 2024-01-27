import numpy as np

def add_intercept(arr):
	x = arr
	if (not isinstance(x, np.ndarray) or len(x.shape) != 2 or len(x[0]) == 0):
		return None
	ones = np.full((x.shape[0], 1), 1)
	return np.concatenate((ones, x), axis=1)

def predict_(x, theta):
	if (not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray) or
	 len(x.shape) != 2 or x.shape[1] != 1 or theta.shape != (2, 1)):
		return None
	return add_intercept(x) @ theta

def loss_elem_(y, y_hat):
	try :
		if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or
		y.shape != y_hat.shape or len(y.shape) != 2 or y.shape[1] != 1):
			return None
		fy_hat = y_hat.astype(float)
		fy = y.astype(float)
		return (fy_hat - fy) ** 2
	except:
		return None


def loss_(y, y_hat):
	try :
		loss = loss_elem_(y, y_hat)
		if (loss is None):
			return
		return loss.sum() / (2 * len(y))
	except:
		return None
