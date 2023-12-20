from Matrix import Matrix

class Vector(Matrix):
	def __init__(self, arg):
		if (isinstance(arg, list)):
			if len(arg) != 1 and (len(arg) == 0 or len(arg[0]) != 1):
				raise ValueError("Error: Not a Vector")
		super().__init__(arg)
	
	def __mul__(self, arg):
		if (isinstance(arg, Vector)):
			if (arg.shape != self.shape):
				raise ValueError("Error: different shape")
		return super().__mul__(arg)

	def dot(self, arg):
		if (not isinstance(arg, Vector)):
				raise ValueError("Error: Invalid arg")
		if (arg.shape != self.shape):
			raise ValueError("Error: Vectors not dotable cause of different shapes")
		return (self.T() * arg)[0][0]
