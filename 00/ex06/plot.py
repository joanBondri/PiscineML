import numpy as np
import matplotlib.pyplot as plt
from prediction import predict_

def plot(x, y, theta):
	plt.scatter(x, y, color="green")
	plt.plot(x, predict_(x, theta), color="red")
	plt.xlabel("X-axis")
	plt.ylabel("Y-axis")
	plt.show()