import math
import numpy as np
def zscore(x):
	if not isinstance(x, np.ndarray) or (len(x.shape) != 1 and len(x.shape) != 2) or (len(x.shape) == 2 and x.shape[1] != 1):
		return
	if (len(x.shape) == 2):
		X = x.T[0]
	else:
		X = x
	m = mean(X)
	s = std(X)
	if (not m or not s):
		return
	if (s == 0):
		return x
	return (X - m) / s


def std (arrDisorder):
	arr = arrDisorder
	lenArr = len(arr)
	if (not isinstance(arr, np.ndarray) or
		not (arr.ndim == 1 and arr.dtype.kind in ('i', 'f')) or lenArr == 0):
		return
	if (lenArr == 1):
		return 0
	meanArr = mean(arr)
	res = 0
	for i in arr:
		res += (i - meanArr) ** 2
	return math.sqrt(res / (lenArr))


def mean(x) :
	if not isinstance(x, np.ndarray) or len(x.shape) != 1:
		return
	len_list = len(x)
	res = 0
	for n in x :
		res += n
	return res / len_list

