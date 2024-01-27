import numpy as np
from tools import add_intercept

class MyLinearRegression():
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        if (isinstance(theta, list)):
            self.theta = np.array(theta)
        else :
            self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter

    def predict_(self, x):
        if (not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray) or
        len(x.shape) != 2 or self.theta.shape[0] != x.shape[1] + 1 or self.theta.shape[1] != 1):
            return None
        X = x.astype(float)
        return add_intercept(X) @ self.theta
            
    def loss_elem_(self, y, y_hat):
        try :
            if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or
            y.shape != y_hat.shape or len(y.shape) != 2 or y.shape[1] != 1):
                return None
            fy_hat = y_hat.astype(float)
            fy = y.astype(float)
            return (fy_hat - fy) ** 2
        except:
            return None

    def loss_(self, y, y_hat):
        if (not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or
        y.shape != y_hat.shape or len(y.shape) != 2 or y.shape[1] != 1 or len(y) == 0):
            return None
        fy_hat = y_hat.astype(float)
        fy = y.astype(float)

        diff = fy_hat - fy
        dot = (diff.T @ diff)[0][0]
        return 1 / (2 * len(y)) * dot
        
    def gradient(self, x, y):
        if (not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray) or not isinstance(y, np.ndarray) or
        len(x.shape) != 2 or len(y.shape) != 2 or
        len(y) == 0 or y.shape[0] != x.shape[0] or y.shape[1] != 1 or
        self.theta.shape[0] != x.shape[1] + 1 or self.theta.shape[1] != 1):
            return None
        X = x.astype(float)
        Y = y.astype(float)
        numberpred = len(X)
        Xp = add_intercept(X)
        res = (Xp.T @ (Xp @ self.theta - Y)) / numberpred
        return res

    def fit_(self, x, y):
        if (not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray) or not isinstance(y, np.ndarray) or
        len(x.shape) != 2 or len(y.shape) != 2 or
        len(y) == 0 or y.shape[0] != x.shape[0] or y.shape[1] != 1 or
        self.theta.shape[0] != x.shape[1] + 1 or self.theta.shape[1] != 1):
            return None
        if not isinstance(self.alpha, float) or not isinstance(self.max_iter, int):
            return None
        if (self.theta.dtype != np.float64):
            self.theta = self.theta.astype(np.float64)
        for i in range(self.max_iter):
            res = self.gradient(x, y)
            if (res is None):
                return None
            self.theta -= self.alpha * res
        return self.theta