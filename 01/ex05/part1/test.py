import numpy as np
from mylinearregression import MyLinearRegression as MyLR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLR([[1.], [1.], [1.], [1.], [1]])
print(f"mylr.predict_(X) = {mylr.predict_(X)}", end = "\n\n")
print(f"mylr.cost_elem_(X,Y) = {mylr.cost_elem_(X,Y)}", end = "\n\n")
print(f"mylr.cost_(X,Y) = {mylr.cost_(X,Y)}", end = "\n\n")
mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=200000)
print(f"mylr.theta = {mylr.theta}", end = "\n\n")
print(f"mylr.predict_(X) = {mylr.predict_(X)}", end = "\n\n")
print(f"mylr.cost_elem_(X,Y) = {mylr.cost_elem_(X,Y)}", end = "\n\n")
print(f"mylr.cost_(X,Y) = {mylr.cost_(X,Y)}", end = "\n\n")