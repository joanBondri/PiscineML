import numpy as np
from tools import is_column_vector, transform_row_vector_to_column_vector, add_intercept


def simple_predict_(X, theta):
    if theta.size == 0 or not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != theta.shape[0]:
        print("Incompatible dimension match between X and theta.")
        return
    return X @ theta