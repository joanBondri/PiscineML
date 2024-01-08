import numpy as np
from tools import is_column_vector, transform_row_vector_to_column_vector, add_intercept

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

    def normalize(self, X):
        X_no_intercept = X[:,1:]
        std = np.std(X_no_intercept, axis=0)
        X_normalized = (X_no_intercept - np.mean(X_no_intercept, axis=0)) / std
        return add_intercept(X_normalized)
    
    def adjust_theta_after_normal_(self, X):
        X_no_intercept = X[:,1:]
        mean_X = np.mean(X_no_intercept, axis=0)
        std_X = np.std(X_no_intercept, axis=0)
        tt = self.theta.T[0]
        tt_cpy = tt
        tt = tt / std_X
        tt[0] = tt_cpy[0] - np.sum((mean_X / std_X) * tt_cpy[1:])
        self.theta = transform_row_vector_to_column_vector(tt)

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
        X_normalize = self.normalize(x)
        for i in range(n_cycle):
            if i % 10000 == 0 :
                print(f"iteration {i}th => costfunction = {self.cost_(X_normalize, y)}")
            res = self.calcule_derivative(X_normalize, y)
            if (res is None):
                return
            self.theta -= alpha * res
        self.adjust_theta_after_normal_(x)
        print(f"theta = {self.theta}")
        return self.theta