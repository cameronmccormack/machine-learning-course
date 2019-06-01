import numpy as np
import scipy.optimize as opt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import ex2

def plotDecisionBoundary(theta, X, y):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((np.size(u), np.size(v)))
    for i in range(1, np.size(u)):
        for j in range(1, np.size(v)):
            z[i, j] = theta.T @ np.array([[1], [u[i]], [v[j]]])
    z = z.T
    plt.contour(u, v, z, [0, 0])
    plt.show()


def mapFeature(X1, X2):
    degree = 6
    out = np.ones((np.size(X1, 0), 1))
    for i in range (1, degree):
        for j in range(0, i):
            out = np.c_[out, (X1**(i-j))*(X2**j)]
    return out

if __name__ == "__main__":

    # Exercise 2 - logistic regression

    # load data
    data = np.loadtxt("data/ex2data2.txt", delimiter=",")
    X, y = data[:, :2], data[:, 2:]

    # plot data

    print("Plotting data with + indicating (y=1) examples and o indicating "
          "(y=0) examples.")
    ex2.plotData(X, y)
    c = input("\nProgram paused. Press enter to continue.")

    # part 1: regularized logistic regression

    X = mapFeature(X[:, 0], X[:, 1])
    initial_theta = np.zeros((np.size(X, 1), 1))
    lambda = 1

    cost, grad = costFunctionReg(initial_theta, X, y, lambda)
