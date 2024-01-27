import numpy as np
from prediction import predict_
x = np.arange(1,6).reshape(-1, 1)
# Example 1:
print("Test 1:")
theta1 = np.array([[5], [0]])
print(predict_(x, theta1))

# Example 2:
print("Test 2:")
theta2 = np.array([[0], [1]])
print(predict_(x, theta2))

# Example 3:
print("Test 3:")
theta3 = np.array([[5], [3]])
print(predict_(x, theta3))

# Example 4:
print("Test 4:")
theta4 = np.array([[-3], [1]])
print(predict_(x, theta4))