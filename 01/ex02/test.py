import numpy as np
from fit import fit_
from pred import predict_

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
theta1 = np.array([[1.], [1.]])
theta1 = fit_(theta1, X1, Y1, alpha = 0.01, n_cycle=2000)
print(f"theta1 = {theta1}", end="\n\n")
print(f"theta1_predict = {predict_(theta1, X1)}", end="\n\n")

X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,80.]])
Y2 = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta2 = np.array([[42.], [1.], [1.], [1.]])
theta2 = fit_(theta2, X2, Y2, alpha = 0.0005, n_cycle=42000)
print(f"theta2 = {theta2}", end="\n\n")
print(f"theta2_predict = {predict_(theta2, X2)}", end="\n\n")
