import math
import numpy as np

def minmax(x):
	if not isinstance(x, np.ndarray) or (len(x.shape) != 1 and len(x.shape) != 2) or (len(x.shape) == 2 and x.shape[1] != 1):
		return
	if (len(x.shape) == 2):
		X = x.T[0]
	else:
		X = x
	min = np.min(X)
	max = np.max(X)
	if (max - min == 0) :
		return X
	return X - min / (max - min)