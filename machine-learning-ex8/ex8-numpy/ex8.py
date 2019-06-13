import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def estimateGaussian(X):
    m = np.size(X, 0)
    mu = (1/m) * np.sum(X, axis=0)
    sigma2 = (1/m) * np.sum((X - mu.T)**2, axis=0)
    return mu, sigma2


def multivariateGaussian(X, mu, Sigma2):
    k = np.size(mu)
    Sigma2 = np.diag(Sigma2)
    X = X - mu.T
    p = (((2*np.pi)**(-k/2)) * (np.det(Sigma2)**-0.5)
         * np.exp(-0.5 * np.sum((X @ np.pinv(Sigma2)) * X, axis=1)))
    return p


if __name__ == "__main__":
    # part 1: load example dataset

    # load and plot data
    print("Vizualizing example dataset for outlier detection.")
    file_data = sio.loadmat("data/ex8data1.mat")
    X = file_data["X"]
    plt.figure(1)
    plt.plot(X[:, 0], X[:, 1], 'bx')
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (mb/s)")
    plt.show(block=False)
    c = input("\nProgram paused. Press enter to continue.")

    # part 2: estimate the dataset statistics

    # estimate mu and sigma2
    mu, sigma2 = estimateGaussian(X)

    # density of the multivariate normal at each data point of X and visualize
    p = multivariateGaussian(X, mu, sigma2)
    #visualizeFit(X, mu, sigma2, 2)
    c = input("\nProgram paused. Press enter to continue.")
