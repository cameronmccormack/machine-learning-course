import numpy as np
import scipy.optimize as opt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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
