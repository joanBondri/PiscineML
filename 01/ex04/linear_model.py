import numpy as np
from mylinearregression import MyLinearRegression as MLR

def main():
    data = np.genfromtxt('are_blue_pills_magics.csv', delimiter=',', skip_header=1, usecols=(1, 2))
    X = np.array(data[:, 0])
    Y = np.array(data[:, 1])
    yop = MLR([[0], [0]])
    

if __name__ == "__main__":
    main()