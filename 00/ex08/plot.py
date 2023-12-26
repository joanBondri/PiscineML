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
	if (not is_row_vector(x) or not is_row_vector(y) or x.shape != y.shape or (not is_row_vector(theta) and theta.shape[0] == 2)):
		return
	plt.scatter(x, y, color="blue")
	plt.plot(x, predict_(x, theta), color="red")
	plt.vlines(x, y, predict_(x, theta), color='green', linestyle='--', label='Vertical Line')
	plt.xlabel("X-axis")
	plt.ylabel("Y-axis")
	plt.show()