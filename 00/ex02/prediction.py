import numpy as np
def simple_predict(x : np.ndarray, theta : np.ndarray):
	if (not (x.ndim == 1 and x.dtype.kind in ('i', 'f')) or
	 	not (theta.ndim == 1 and theta.dtype.kind in ('i', 'f') and len(theta) == 2)):
		raise TypeError("Error: wrong arg type")
	resList = [theta[0] + theta[1] * xi for xi in x]
	return np.array(resList)