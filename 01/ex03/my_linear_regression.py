import numpy as np
from tools import add_intercept

class MyLinearRegression():
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        newalpha = alpha
        self.alpha = newalpha

        newmax_iter = max_iter
        self.max_iter = newmax_iter

        newthetas = thetas
        self.thetas = newthetas

    def predict_(self, x):
        try :
            if (not isinstance(x, np.ndarray) or
            len(x.shape) != 2 or x.shape[1] != 1):
                return None
            return add_intercept(x) @ self.thetas
        except:
            return None
        
    
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
        try :
            loss = self.loss_elem_(y, y_hat)
            if (loss is None):
                return
            return loss.sum() / (2 * len(y))
        except:
            return None
    
    def calcule_derivative(self, x, y):
        try :
            numberpred = len(x)
            X = add_intercept(x)
            res = (X.T @ (X @ self.thetas - y)) / numberpred
            return res
        except:
            return None

    def fit_(self, x, y):
        try :
            if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or
            len(x.shape) != 2 or x.shape[1] != 1 or y.shape != x.shape):
                return None
            if (self.thetas.dtype != np.float64):
                self.thetas = self.thetas.astype(np.float64)
            if not isinstance(self.alpha, float) or not isinstance(self.max_iter, int):
                return None
            for i in range(self.max_iter):
                res = self.calcule_derivative(x, y)
                if (res is None):
                    return None
                self.thetas -= self.alpha * res
            return self.thetas
        except:
            return None