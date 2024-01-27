import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import sys
from my_linear_regression import MyLinearRegression as MyLR
from plot import plot
from tools import predict_, loss_


def show_plot(x, y, theta, one_x):
    plot(x, y, theta, b_legend=True,
        axes_labels=["Quantity of blue pill (in micrograms)",
                    "Space driving score"],
        data_labels={"raw": r"$S_{true}$(pills)",
                    "prediction": r"$S_{predict}$(pills)"})
    


def print_predict(datafile="are_blue_pills_magics.csv"):
    try:
        data = pd.read_csv(datafile)
    except:
        print("An error occured during the reading of the dataset.")
        sys.exit()

    # Checking the dataset:
    cols = data.columns
    if not all([c in ["Patient", "Micrograms", "Score"] for c in cols]):
        print("Unexpected column in the dataset.")
        sys.exit()

    try:
        # Definition of x and y:
        x = data.Micrograms.values.reshape(-1, 1)
        y = data.Score.values.reshape(-1, 1)

        # Model and training
        thetas = np.random.rand(2, 1)
        mylr = MyLR(thetas, .05, 10000)
        print(f"mylr.fit_(x, y) = {mylr.fit_(x, y)}")
        # Plot
        plot(x, y, mylr.theta, b_legend=True,
             axes_labels=["Quantity of blue pill (in micrograms)",
                          "Space driving score"],
             data_labels={"raw": r"$S_{true}$(pills)",
                          "prediction": r"$S_{predict}$(pills)"})
        return x, y
    except:
        print("Error")
        sys.exit()


def show_thetas_losses(x, y):
    n = 4
    theta0 = np.linspace(80, 96, n)
    theta1 = np.linspace(-14, -4, 100)

    PiYG = get_cmap('PiYG', n)
    fig, axe = plt.subplots(1, 1, figsize=(15, 10))
    for t0, color in zip(theta0, PiYG(range(n))):
        losses = []
        for t1 in theta1:
            ypred = predict_(x, np.array([[t0], [t1]]))
            losses.append(loss_(y, ypred))
        axe.plot(theta1, np.array(losses),
                 label=r"J($\theta_0$ = " + f"{t0}," + r"$\theta_1$)",
                 lw=2.5,
                 c=color)
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"cost function J($\theta_0 , \theta_1$)")
    axe.set_ylim([10, 150])
    plt.show()

if __name__ == '__main__':
    x, y = print_predict()
    show_thetas_losses(x, y)