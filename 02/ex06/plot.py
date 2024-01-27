import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, theta, b_legend = True,axes_labels = ["x", "y"], data_labels = {"raw":"raw", "prediction":"prediction"}):
	try:
		fig, axes = plt.subplots(1,1, figsize=(10,8))
		axes.scatter(x, y, label = data_labels['raw'], c='#101214')
		axes.scatter(x, predict_(x, theta), label = data_labels['prediction'], c='#4287f5')
		plt.legend()
		plt.xlabel(axes_labels[0])
		plt.ylabel(axes_labels[1])
		if b_legend:
			plt.legend()
		plt.grid()
		plt.show()
	except:
		return None