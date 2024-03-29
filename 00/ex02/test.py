from prediction import simple_predict
import numpy as np
x = np.arange(1,6).reshape(-1, 1)
theta1 = np.array([[5], [0]])

print("1st test")
print(simple_predict(x, theta1))
theta2 = np.array([[0], [1]])

print("2nd test")
print(simple_predict(x, theta2))
theta3 = np.array([[5], [3]])

print("3rd test")
print(simple_predict(x, theta3))
theta4 = np.array([[-3], [1]])

print("4th test")
print(simple_predict(x, theta4))