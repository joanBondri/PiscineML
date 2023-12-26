import numpy as np
from tools import add_intercept

def predict_(x : np.ndarray, theta : np.ndarray):
	if (not (x.ndim == 2 and x.dtype.kind in ('i', 'f') and x.shape[1] == 1) or
	 	not (theta.ndim == 2 and theta.dtype.kind in ('i', 'f') and len(theta) == 2)):
		raise TypeError("Error: wrong arg type")
	return np.sum(add_intercept(x) * theta.reshape(1, -1), axis=1)