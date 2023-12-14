
class Matrix :
	def __init__(self, arg):
		if (isinstance(arg, tuple) and arg.length == 2):
			self.data = [[0] * arg[0]] * arg[1]
			self.shape = arg
		elif (isinstance(arg, list)):
			if arg.length == 0:
				raise ValueError("Error: Not a Matrix")
			for i in arg:
				if i.length != arg[0].length:
					raise ValueError("Error: Not a Matrix")
			self.data = arg
			self.shape = (arg.length, arg[0].length)
		else:
			raise ValueError("Error: Invalid value")
	def __add__(self, arg) :
		if (not isinstance(arg, Matrix)):
			raise ValueError("Error: Invalid arg")
		if (arg.shape != self.shape):
			raise ValueError("Error: different shape")
		res = [[self.data[row][column] + arg[row][column] for column in range(len(self.data[row]))] for row in range(len(self.data))]
		return Matrix(res)
	def __radd__(self, arg) :
		return self.__add__(arg)

	def __sub__(self, arg) :
		if (not isinstance(arg, Matrix)):
			raise ValueError("Error: Invalid arg")
		if (arg.shape != self.shape):
			raise ValueError("Error: different shape")
		res = [[self.data[row][column] - arg[row][column] for column in range(len(self.data[row]))] for row in range(len(self.data))]
		return Matrix(res)
	
	def __rsub__(self, arg) :
		return self.__sub__(arg)
	
	def __truediv__(self, arg) :
		if (not isinstance(arg, (int, float))):
			raise ValueError("Error: Invalid arg")
		if (arg == 0):
			raise ValueError("Error: division by zero")
		res = [[self.data[row][column] / arg for column in range(len(self.data[row]))] for row in range(len(self.data))]
		return Matrix(res)

	def __rtruediv__(self, arg):
		return self.__truediv__(arg)
	

