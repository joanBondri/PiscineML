class Matrix :
	def __init__(self, arg):
		if (isinstance(arg, tuple) and len(arg) == 2):
			self.data = [[0] * arg[0]] * arg[1]
			self.shape = arg
		elif (isinstance(arg, list)):
			if len(arg) == 0:
				raise ValueError("Error: Not a Matrix")
			for i in arg:
				if len(i) == 0 or len(i) != len(arg[0]):
					raise ValueError("Error: Not a Matrix")
			self.data = arg
			self.shape = (len(arg), len(arg[0]))
		else:
			raise ValueError("Error: Invalid value")

	def __add__(self, arg) :
		if (not isinstance(arg, Matrix)):
			raise ValueError("Error: Invalid arg")
		if (arg.shape != self.shape):
			raise ValueError("Error: different shape")
		res = [[self.data[row][column] + arg.data[row][column] for column in range(len(self.data[row]))] for row in range(len(self.data))]
		print(res)
		return Matrix(res)
	def __radd__(self, arg) :
		return self.__add__(arg)

	def __sub__(self, arg) :
		if (not isinstance(arg, Matrix)):
			raise ValueError("Error: Invalid arg")
		if (arg.shape != self.shape):
			raise ValueError("Error: different shape")
		res = [[self.data[row][column] - arg.data[row][column] for column in range(len(self.data[row]))] for row in range(len(self.data))]
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
	
	def __mul__(self, arg):
		if (isinstance(arg, (int, float))):
			res = [[self.data[row][column] * arg for column in range(len(self.data[row]))] for row in range(len(self.data))]
			return Matrix(res)
		if (isinstance(arg, Matrix)):
			if (self.shape[1] != arg.shape[0]):
				raise ValueError("Error: shapes not multiplicative")
			len_adding = self.shape[1]
			res = [[0] * arg.shape[1] for _ in range(self.shape[0])]
			for i in range(self.shape[0]):
				for j in range(arg.shape[1]):
					for n in range(len_adding):
						res[i][j] += self.data[i][n] * arg.data[n][j]
			return Matrix(res)
		raise ValueError("Error: Invalid arg")
		
	def __getitem__(self, index):
		return self.data[index]

	def __rmul__(self, arg):
		if (isinstance(arg, (int, float))):
			return self.__mul__(arg)
		return arg.__mul__(self)

	def __str__(self):
		return "Matrix(" + str(self.data) + ")"
	
	def T(self):
		if self.shape[0] == 0 or self.shape[0] == 1:
			return Matrix(self.shape)
		res = [[self.data[row][column] for row in range(len(self.data))] for column in range(len(self.data[0]))]
		return Matrix(res)
	def __repr__(self):
		return "Matrix:\n\tshape : " + str(self.shape) + "\n\t "+ str(self.data) + ")"