import numpy as np

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
