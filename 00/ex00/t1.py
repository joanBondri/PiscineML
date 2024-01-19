from Matrix import Vector, Matrix

liste = [[1, 2, 3, 4, 5]]
vector1 = Vector(liste)
vector2 = Vector([[2, 3, 4, 5, 6]])
print(vector1.shape)
print(vector2.shape)
print(f" res {vector1.dot(vector2)}" )
print(Matrix((2, 3)))