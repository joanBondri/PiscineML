import numpy as np
from my_linear_regression import MyLinearRegression as MyLR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLR([[1.], [1.], [1.], [1.], [1.]])

y_hat = mylr.predict_(X)

print("test1")
print(y_hat)

print("\n\ntest2")
print(mylr.loss_elem_(Y, y_hat))

print("\n\ntest3")
print(mylr.loss_(Y, y_hat))

mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)

print("\n\ntest3 -> fit :")
print(mylr.theta)

y_hat = mylr.predict_(X)

print("\n\ntest4 -> predict :")
print(y_hat)

print("\n\ntest5 -> loss_elem :")
print(mylr.loss_elem_(Y, y_hat))

print("\n\n test6 -> loss:")
print(mylr.loss_(Y, y_hat))