import numpy as np
import math

class TinyStatistician :
	def __init__(self):
		pass
	def mean(self, x) :
		if not isinstance(x, list) and not isinstance(x, np.ndarray):
			raise TypeError("Not good type")
		len_list = len(x)
		res = 0
		for n in x :
			res += n
		return res / len_list

	def median(self, arr):
		return self.percentile(arr, 50)

	def quartile(self, arr):
		return [self.percentile(arr, 25), self.percentile(arr, 75)]

	def percentile (self, arrDisorder, p : int):
		if (isinstance(arrDisorder, list)):
			arr = np.array(arrDisorder)
		else:
			arr = arrDisorder
		lenArr = len(arr)
		if (p > 100 or p < 0 or
	  		not isinstance(arr, np.ndarray) or
	  		not (arr.ndim == 1 and arr.dtype.kind in ('i', 'f')) or
			lenArr == 0):
			return
		arr.sort()
		percentilePosition = (p / 100) * (lenArr - 1)
		decimal = percentilePosition - math.floor(percentilePosition)
		lowerValue = arr[int(percentilePosition - decimal)]
		# print(f"lowervalue : {lowerValue}")
		if (decimal == 0.0):
			return lowerValue
		return lowerValue + decimal * (arr[int(percentilePosition - decimal) + 1] - lowerValue)

	def var (self, arrDisorder):
		if (isinstance(arrDisorder, list)):
			arr = np.array(arrDisorder)
		else:
			arr = arrDisorder
		lenArr = len(arr)
		if (not isinstance(arr, np.ndarray) or
			not (arr.ndim == 1 and arr.dtype.kind in ('i', 'f')) or
			lenArr == 0):
			return
		if (lenArr == 1):
			return 0
		mean = self.mean(arr)
		res = 0
		for i in arr:
			res += (i - mean) ** 2
		return res / (lenArr - 1)

	def std(self, arr):
		return math.sqrt(self.var(arr))
