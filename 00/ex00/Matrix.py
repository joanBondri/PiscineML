class Matrix :
	def __init__(self, *args):
		if len(args) != 1:
			raise ValueError("No or multiple arguments given. Only one is expected")
		self.checking_data(*args)
		self.data = self._get_data_(*args)
		self.shape = self._get_shape_(*args)


	@staticmethod
	def checking_data(arg):
		if not isinstance(arg, (tuple, list)):
			raise TypeError("Unexpected data type, must be a list or a tuple.")
		if isinstance(arg, tuple):
			l_t = len(arg)
			if l_t != 2:
				raise ValueError("Unexpected length for shape argument.")
			if not isinstance(arg[0], (int, float)) or not isinstance(arg[1], (int, float)):
				raise TypeError("Unexpected type within shape argument")
		if isinstance(arg, list):
			if any([not isinstance(l, list) for l in arg]):
				raise TypeError("Unexpected type within the data argument.")
			l_t = len(arg)
			if l_t == 0:
				raise ValueError("Unexpected length for data argument.")
			for l_line in arg:
				if any([not isinstance(elem, (int, float)) for elem in l_line]):
					raise TypeError("Unexpected element type for data argument.")
			l_ref = len(arg[0])
			if any([len(line) != l_ref for line in arg]):
				raise ValueError("Heterogenous column length in data argument.")

	def _get_data_(self, arg):
		if isinstance(arg, list):
			dupplicate = arg.copy()
			return dupplicate
		else:
			# arg is a tuple
			n_line, n_col = arg[0], arg[1]
			data = []
			[data.append([0] * n_col) for _ in range(n_line)]
			return data


	def _get_shape_(self, arg):
		if isinstance(arg, tuple):
			return arg
		else:
			n_line, n_col = len(arg), len(arg[0])
			return (n_line, n_col)


	def __add__(self, arg) :
		if (not isinstance(arg, Matrix)):
			raise ValueError("Error: Invalid arg")
		if (arg.shape != self.shape):
			raise ValueError("Error: different shape")
		res = [[self.data[row][column] + arg.data[row][column] for column in range(len(self.data[row]))] for row in range(len(self.data))]
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
		if (not isinstance(arg, Matrix)):
			raise ValueError("Error: Invalid arg")
		if (arg.shape != self.shape):
			raise ValueError("Error: different shape")
		res = [[arg.data[row][column] - self.data[row][column] for column in range(len(self.data[row]))] for row in range(len(self.data))]
		return Matrix(res)
	
	def __truediv__(self, arg) :
		if (not isinstance(arg, (int, float))):
			raise ValueError("Error: Invalid arg")
		if (arg == 0):
			raise ValueError("Error: division by zero")
		res = [[self.data[row][column] / arg for column in range(len(self.data[row]))] for row in range(len(self.data))]
		return Matrix(res)

	def __rtruediv__(self, arg):
		if (not isinstance(arg, (int, float))):
			raise ValueError("Error: Invalid arg")
		control = [any([self.data[row][column] == 0 for column in range(len(self.data[row]))]) for row in range(len(self.data))]
		if (any(control)):
			raise ValueError("Error: division by zero")
		res = [[arg / self.data[row][column] for column in range(len(self.data[row]))] for row in range(len(self.data))]
		return Matrix(res)
	
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
		elif isinstance(arg, Vector):
			if self.shape[1] == arg.shape[0]:
				res = Vector(arg.shape)
				for ii in range(self.shape[0]):
					for jj in range(self.shape[1]):
						res.data[ii][0] = self.data[ii][jj] * arg[jj][0]
				return res
		raise ValueError("Error: Invalid arg")

	def __rmul__(self, arg):
		return self.__mul__(arg)

	def __getitem__(self, index):
		return self.data[index]

	def __repr__(self):
		try:
			radical = "Matrix(["
			end = "])"
			for line in self.data:
				radical += str(line) + ' '
			return radical[:-1] + end
		except:
			raise AttributeError("Something wrong happened")

	def __str__(self):
		return self.__repr__()
	
	def T(self):
		new = Matrix(self.shape[::-1])
		for ii in range(self.shape[0]):
			for jj in range(self.shape[1]):
				new.data[jj][ii] = self.data[ii][jj]
		return new
	
class Vector(Matrix):
	def __init__(self, arg):
		super().__init__(arg)
		if all(l != 1 for l in self.shape):
			raise ValueError("Wrong shape")

	def T(self):
		new = Vector(self.shape[::-1])
		for ii in range(self.shape[0]):
			for jj in range(self.shape[1]):
				new.data[jj][ii] = self.data[ii][jj]
		return new


	def dot(self, v):
		if not isinstance(v, Vector) or self.shape != v.shape:
			raise TypeError("Wrong argument")
		res = 0
		for ii in range(self.shape[1]):
			res += self.data[0][ii] * v.data[0][ii]
		return res
		
	def __add__(self, arg):
		if (not isinstance(arg, Vector)):
			raise ValueError("Error: Invalid arg")
		if (arg.shape != self.shape):
			raise ValueError("Error: different shape")
		res = [[self.data[row][column] + arg.data[row][column] for column in range(len(self.data[row]))] for row in range(len(self.data))]
		return Vector(res)
	  

	def __radd__(self, arg):
		return self.__add__(arg)
	
	def __sub__(self, arg):
		if (not isinstance(arg, Vector)):
			raise ValueError("Error: Invalid arg")
		if (arg.shape != self.shape):
			raise ValueError("Error: different shape")
		res = [[self.data[row][column] - arg.data[row][column] for column in range(len(self.data[row]))] for row in range(len(self.data))]
		return Vector(res)


	def __rsub__(self, arg):
		if (not isinstance(arg, Vector)):
			raise ValueError("Error: Invalid arg")
		if (arg.shape != self.shape):
			raise ValueError("Error: different shape")
		res = [[arg.data[row][column] - self.data[row][column] for column in range(len(self.data[row]))] for row in range(len(self.data))]
		return Vector(res)

	
	def __truediv__(self, arg):
		if (not isinstance(arg, (int, float))):
			raise ValueError("Error: Invalid arg")
		if (arg == 0):
			raise ValueError("Error: division by zero")
		res = [[self.data[row][column] / arg for column in range(len(self.data[row]))] for row in range(len(self.data))]
		return Vector(res)


	def __rtruediv__(self, arg):
		if (not isinstance(arg, (int, float))):
			raise ValueError("Error: Invalid arg")
		control = [any([self.data[row][column] == 0 for column in range(len(self.data[row]))]) for row in range(len(self.data))]
		if (any(control)):
			raise ValueError("Error: division by zero") 
		res = [[arg / self.data[row][column] for column in range(len(self.data[row]))] for row in range(len(self.data))]
		return Vector(res)

	def __mul__(self, other):
		if not isinstance(other, (Matrix, int, float, Vector)):
			raise ArithmeticError("wrong type of arg")
		if isinstance(other, (int, float)):
			res = Vector(self.shape)
			for ii in range(self.shape[0]):
				for jj in range(self.shape[1]):
					res.data[ii][jj] = self.data[ii][jj] * other
			return res
		elif isinstance(other, Matrix):
			if self.shape[1] == other.shape[0]:
				res = Vector((self.shape[0], other.shape[1]))
				for ii in range(self.shape[0]):
					for kk in range(other.shape[1]):
						for jj in range(self.shape[1]):
							res.data[ii][kk] += self.data[ii][jj] * other.data[jj][kk]
				return res
			else:
				raise ArithmeticError("Mismatch dimension between Vector instances.")
		elif isinstance(other, Vector):
			if self.shape[1] == other.shape[0]:
				res = Vector(other.shape)
				for ii in range(self.shape[0]):
					for jj in range(self.shape[1]):
						res.data[ii][0] += self.data[ii][jj] * other.data[jj][0]
				return res
			else:
				raise ArithmeticError("Mismatch dimension between Matrix instances.")


	def __rmul__(self, other):
		if not isinstance(other, (int, float, Vector, Matrix)):
			raise ArithmeticError("wrong type of arg")
		if isinstance(other, (int, float)):
			res = Vector(self.shape)
			for ii in range(self.shape[0]):
				for jj in range(self.shape[1]):
					res.data[ii][jj] = self.data[ii][jj] * other
			return res
		elif isinstance(other, Vector):
			if other.shape[1] == self.shape[0]:
				res = Vector((other.shape[0], self.shape[1]))
				for ii in range(other.shape[0]):
					for kk in range(self.shape[1]):
						for jj in range(other.shape[1]):
							res.data[ii][kk] += other.data[ii][jj] * self.data[jj][kk]
				return res
			else:
				raise ArithmeticError("Mismatch dimension between Vector instances.")
		elif isinstance(other, Matrix):
			if other.shape[1] == self.shape[0]:
				res = Vector((other.shape[0], self.shape[1]))
				for ii in range(other.shape[0]):
					for jj in range(other.shape[1]):
						res.data[ii][0] += other.data[ii][jj] * self.data[jj][0]
				return res
			else:
				raise ArithmeticError("Mismatch dimension between Vector instances.")

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		radical = "Vector(["
		end = "])"
		for line in self.data:
			radical += str(line) + ' '
		return radical[:-1] + end
	
