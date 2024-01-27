from minmax import minmax
import numpy as np

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(minmax(X))


X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
print(minmax(X))

Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
print(minmax(Y))