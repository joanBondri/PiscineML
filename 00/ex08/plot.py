import numpy as np
import matplotlib.pyplot as plt
from prediction import predict_
from tools import is_row_vector

def plot(x, y, theta):
	plt.scatter(x, y, color="green")
	plt.plot(x, predict_(x, theta), color="red")
	plt.xlabel("X-axis")
	plt.ylabel("Y-axis")
	plt.show()

def plot_with_loss(x, y, theta):
    try:
        _, axe = plt.subplots(1, 1, figsize = (15,8))
        axe.scatter(x, y, c="royalblue")
        ypred = predict_(x, theta)
        axe.plot(x, ypred, '-', c='darkorange')

        # Generator for the residual segments
        g_dist = ([np.array([xi, xi]), np.array([yi, ypredi])] for xi, yi, ypredi in zip(x, y, ypred))

        for residual_i in g_dist:
            axe.plot(residual_i[0], residual_i[1], '--', c='red')
        plt.show()
        print(f"x = {x}, y = {y}, theta= {theta}")
    except:
        print("Please check the dimension of the different arguments.", file=sys.stderr)
        return None