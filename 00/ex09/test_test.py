import numpy as np
import warnings
warnings.filterwarnings("ignore")
from other_losses import mse_, r2score_, rmse_, mae_

X = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y = np.array([[2], [2], [2], [2], [2], [2], [2]])

# Test case 1: X and Y have different shapes
X1 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y1 = np.array([[2], [2], [2], [2], [2], [2]])
print("Testing mse_")
print(mse_(X1, Y1))

# Test case 2: X and Y have different values
X2 = np.array([[1], [2], [3], [4], [5], [6], [7]])
Y2 = np.array([[2], [4], [6], [8], [10], [12], [14]])
print("Testing mse_")
print(mse_(X2, Y2))

# Test case 3: X and Y have the same values
X3 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y3 = np.array([[1], [1], [1], [1], [1], [1], [1]])
print("Testing mse_")
print(mse_(X3, Y3))

# Test case 4: X and Y have different shapes
X4 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y4 = np.array([[2], [2], [2], [2], [2], [2]])
print("Testing r2score_")
print(r2score_(X4, Y4))

# Test case 5: X and Y have different values
X5 = np.array([[1], [2], [3], [4], [5], [6], [7]])
Y5 = np.array([[2], [4], [6], [8], [10], [12], [14]])
print("Testing r2score_")
print(r2score_(X5, Y5))

# Test case 6: X and Y have the same values
X6 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y6 = np.array([[1], [1], [1], [1], [1], [1], [1]])
print("Testing r2score_")
print(r2score_(X6, Y6))

# Test case 7: X and Y have different shapes
X7 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y7 = np.array([[2], [2], [2], [2], [2], [2]])
print("Testing rmse_")
print(rmse_(X7, Y7))

# Test case 8: X and Y have different values
X8 = np.array([[1], [2], [3], [4], [5], [6], [7]])
Y8 = np.array([[2], [4], [6], [8], [10], [12], [14]])
print("Testing rmse_")
print(rmse_(X8, Y8))

# Test case 9: X and Y have the same values
X9 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y9 = np.array([[1], [1], [1], [1], [1], [1], [1]])
print("Testing rmse_")
print(rmse_(X9, Y9))

# Test case 10: X and Y have different shapes
X10 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y10 = np.array([[2], [2], [2], [2], [2], [2]])
print("Testing mae_")
print(mae_(X10, Y10))

# Test case 11: X and Y have different values
X11 = np.array([[1], [2], [3], [4], [5], [6], [7]])
Y11 = np.array([[2], [4], [6], [8], [10], [12], [14]])
print("Testing mae_")
print(mae_(X11, Y11))

# Test case 12: X and Y have the same values
X12 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y12 = np.array([[1], [1], [1], [1], [1], [1], [1]])
print("Testing mae_")
print(mae_(X12, Y12))