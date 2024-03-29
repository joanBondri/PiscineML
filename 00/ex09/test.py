import numpy as np
import warnings
warnings.filterwarnings("ignore")
from other_losses import mse_, r2score_, rmse_, mae_

X = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y = np.array([[2], [2], [2], [2], [2], [2], [2]])
print("Testing mse_")
print(mse_(X, Y))

print("Testing r2score_")
print(r2score_(X, Y))

print("Testing rmse_")
print(rmse_(X, Y))

print("Testing mae_")
print(mae_(X, Y))

# Additional tests for mse_
X1 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y1 = np.array([[2], [2], [2], [2], [2], [2], [2]])
print("Additional test for mse_")
print(mse_(X1, Y1))

X2 = np.array([[1], [2], [3], [4], [5], [6], [7]])
Y2 = np.array([[2], [4], [6], [8], [10], [12], [14]])
print("Additional test for mse_")
print(mse_(X2, Y2))

X3 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y3 = np.array([[1], [1], [1], [1], [1], [1], [1]])
print("Additional test for mse_")
print(mse_(X3, Y3))

# Additional tests for r2score_
X4 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y4 = np.array([[2], [2], [2], [2], [2], [2]])
print("Additional test for r2score_")
print(r2score_(X4, Y4))

X5 = np.array([[1], [2], [3], [4], [5], [6], [7]])
Y5 = np.array([[2], [4], [6], [8], [10], [12], [14]])
print("Additional test for r2score_")
print(r2score_(X5, Y5))

X6 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y6 = np.array([[1], [1], [1], [1], [1], [1], [1]])
print("Additional test for r2score_")
print(r2score_(X6, Y6))

# Additional tests for rmse_
X7 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y7 = np.array([[2], [2], [2], [2], [2], [2]])
print("Additional test for rmse_")
print(rmse_(X7, Y7))

X8 = np.array([[1], [2], [3], [4], [5], [6], [7]])
Y8 = np.array([[2], [4], [6], [8], [10], [12], [14]])
print("Additional test for rmse_")
print(rmse_(X8, Y8))

X9 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y9 = np.array([[1], [1], [1], [1], [1], [1], [1]])
print("Additional test for rmse_")
print(rmse_(X9, Y9))

# Additional tests for mae_
X10 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y10 = np.array([[2], [2], [2], [2], [2], [2]])
print("Additional test for mae_")
print(mae_(X10, Y10))

X11 = np.array([[1], [2], [3], [4], [5], [6], [7]])
Y11 = np.array([[2], [4], [6], [8], [10], [12], [14]])
print("Additional test for mae_")
print(mae_(X11, Y11))

X12 = np.array([[1], [1], [1], [1], [1], [1], [1]])
Y12 = np.array([[1], [1], [1], [1], [1], [1], [1]])
print("Additional test for mae_")
print(mae_(X12, Y12))