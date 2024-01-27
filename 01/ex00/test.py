import numpy as np
from pred import predict_
from gradient import simple_gradient

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
Y1 = np.array([[12], [14], [50], [5], [43]])
theta1 = np.array([[2.], [4.]])
print(simple_gradient(X1, Y1, theta1))

x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

theta1 = np.array([2, 0.7]).reshape((-1, 1))
print("test1")
print(simple_gradient(x, y, theta1))

theta2 = np.array([1, -0.4]).reshape((-1, 1))
print("test2")
print(simple_gradient(x, y, theta2))