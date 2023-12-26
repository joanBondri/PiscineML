import numpy as np
from tools import is_column_vector, transform_row_vector_to_column_vector

class MyLinearRegression():
    def __init__(self, theta):
        if isinstance(theta, list):
            theta_temp = np.array(theta)
        if not is_column_vector(theta_temp):
            self.theta = np.array([[]])
        else :
            self.theta = np.array(theta_temp)

    def predict_(self, X):
        if self.theta.size == 0 or not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != self.theta.shape[0]:
            print("Incompatible dimension match between X and theta.")
            return
        return X @ self.theta

    def cost_elem_(self, X, Y):
        prediction = self.predict_(X)
        if prediction is None or not is_column_vector(Y) or X.shape[0] != Y.shape[0] :
            return
        diff = prediction - Y
        return diff ** 2 / (2 * Y.shape[0])

    def cost_(self, X, Y):
        cost_elem = self.cost_elem_(X, Y)
        if (cost_elem is None):
            return
        return cost_elem.sum()
    
    def calcule_derivative(self, x, y):
        prediction = self.predict_(x)
        if prediction is None or not is_column_vector(y) or x.shape[0] != y.shape[0] :
            return
        diff = prediction - y
        derivative_sum = np.sum((diff * x) / len(y), axis=0)
        return transform_row_vector_to_column_vector(derivative_sum)

    def fit_(self, x, y, alpha=.1, n_cycle=10000):
        for i in range(n_cycle):
            if i % 100000 == 0 :
                print(f"iteration {i}th => costfunction = {self.cost_(x, y)}")
            res = self.calcule_derivative(x, y)
            if (res is None):
                return
            self.theta -= alpha * res
        return self.theta