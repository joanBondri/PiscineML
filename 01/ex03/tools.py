import numpy as np

def is_vector(arr):
    return isinstance(arr, np.ndarray) and arr.ndim == 1 or (arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1))

def is_row_vector(arr):
	return isinstance(arr, np.ndarray) and arr.ndim == 1

def is_column_vector(arr):
	return isinstance(arr, np.ndarray) and arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1)

def transform_row_vector_to_column_vector(arr):
	if not is_vector(arr):
		return
	newArr = arr
	if newArr.ndim == 1 or newArr.shape[0] == 1:
		return newArr.reshape(-1, 1)
	return newArr

def transform_column_vector_to_row_vector(arr):
	if not is_vector(arr):
		return
	newArr = arr
	if newArr.ndim == 2 and newArr.shape[0] == 1:
		return newArr[0]
	elif newArr.ndim == 2 and newArr.shape[0] != 1:
		return newArr.reshape(1, -1)[0]
	return newArr

def add_intercept(arr : np.ndarray):
	x = arr
	lenX = len(x)
	if (x.ndim > 2):
		return
	if (x.ndim == 1):
		x = x.reshape(-1, 1)
	ones = np.full((x.shape[0], 1), 1)
	return np.concatenate((ones, x), axis=1)