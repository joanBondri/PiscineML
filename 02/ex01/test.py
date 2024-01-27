import numpy as np
from prediction import predict_

x = np.arange(1,13).reshape((4,-1))

theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
print("test1")
print(predict_(x, theta1))

print("\n\ntest2")
theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
print(predict_(x, theta2))

print("\n\ntest3")
theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
print(predict_(x, theta3))

print("\n\ntest4")
theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
print(predict_(x, theta4))
