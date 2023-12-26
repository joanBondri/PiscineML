import numpy as np
from tools import add_intercept, transform_row_vector_to_column_vector
import numpy as np
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MLR

def plot_basic_linear_regression(theta, X, Y, title="title", x_axis="x_axis", y_axis="y_axis"):
    alpha = 10.0
    res = 1
    while res > 0:
        alpha /= 10
        mlr_temp = MLR(theta)
        res = mlr_temp.cost_(X, Y)
        mlr_temp.fit_(X, Y, alpha=alpha, n_cycle=2)
        cost = mlr_temp.cost_(X, Y)
        if cost is None or res is None:
            print(f"souci, cost = {cost}, res = {res}")
            return
        res = mlr_temp.cost_(X, Y) - res
    mlr = MLR(theta)
    mlr.fit_(X, Y, alpha=alpha, n_cycle=1000000)
    data_x = X[:,1]
    plt.scatter(data_x, Y, color="green")
    plt.scatter(data_x, mlr.predict_(X), color="red")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()
    return

def main():
    path = '../spacecraft_data.csv'
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    X_1 = add_intercept(transform_row_vector_to_column_vector(np.array(data[:, 0])))
    Y_1 = transform_row_vector_to_column_vector(np.array(data[:, 3]))
    plot_basic_linear_regression([[0.0], [0.0]], X_1, Y_1)
    X_1 = add_intercept(transform_row_vector_to_column_vector(np.array(data[:, 2])))
    plot_basic_linear_regression([[0.0], [0.0]], X_1, Y_1)
    return

if __name__ == "__main__":
    main()