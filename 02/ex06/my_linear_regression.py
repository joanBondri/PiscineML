import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tools import add_intercept

class MyLinearRegression():
    def mse_(self, y, y_hat):
        try:
            fy_hat = y_hat.astype(float)
            fy = y.astype(float)
            diff = fy_hat - fy
            vec_res = diff.T @ diff
            return float(1 / (len(y)) * vec_res[0][0])
        except:
            return None

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        if (isinstance(theta, list)):
            self.theta = np.array(theta)
        else :
            self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter

    def predict_(self, x):
        print(f"theta = {self.theta.shape}")
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
    
    def plot(self, x, y, x_for_plot=None, b_legend = True,axes_labels = ["x", "y"], data_labels = {"raw":"raw", "prediction":"prediction"}):
        try:
            if x_for_plot :
                X = x_for_plot
            else :
                X = x
            fig, axes = plt.subplots(1,1, figsize=(10,8))
            axes.scatter(X, y, label = data_labels['raw'], c='#101214')
            axes.scatter(X, self.predict_(x), label = data_labels['prediction'], c='#4287f5')
            plt.legend()
            plt.xlabel(axes_labels[0])
            plt.ylabel(axes_labels[1])
            if b_legend:
                plt.legend()
            plt.grid()
            plt.show()
        except:
            return None

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