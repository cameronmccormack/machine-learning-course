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
    p = (((2*np.pi)**(-k/2)) * (np.linalg.det(Sigma2)**-0.5)
         * np.exp(-0.5 * np.sum((X @ np.linalg.pinv(Sigma2)) * X, axis=1)))
    return p


def visualizeFit(X, mu, sigma2, fig):
    X1, X2 = np.meshgrid(np.linspace(0, 35, 70), np.linspace(0, 35, 70))
    Z = multivariateGaussian(np.c_[np.reshape(X1, (70**2, 1)),
                                   np.reshape(X2, (70**2, 1))],
                             mu, sigma2)
    Z = np.reshape(Z, (np.size(X1, 0), np.size(X1, 1)))
    if np.sum(np.isinf(Z)) == 0:
        ax = fig.gca()
        ax.contour(X1, X2, Z, np.logspace(-20, 0, 20))
        plt.show(block=False)


def selectThreshold(yval, pval):
    # unroll to common shape
    yval = np.reshape(yval, np.size(yval))
    pval = np.reshape(pval, np.size(pval))
    print(yval)
    print(pval)
    steps = np.linspace(np.min(pval), np.max(pval), 1000)
    bestF1 = 0
    for epsilon in steps:
        tp = np.sum(pval[yval == 1] < epsilon)
        fp = np.sum(pval[yval == 0] < epsilon)
        fn = np.sum(pval[yval == 1] >= epsilon)

        prec = tp/(tp+fp) if (tp+fp) > 0 else 0
        rec = tp/(tp+fn) if (tp+fn) > 0 else 0

        F1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1


if __name__ == "__main__":
    # part 1: load example dataset

    # load and plot data
    print("Vizualizing example dataset for outlier detection.")
    file_data = sio.loadmat("data/ex8data1.mat")
    X = file_data["X"]
    fig1 = plt.figure()
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
    visualizeFit(X, mu, sigma2, fig1)
    c = input("\nProgram paused. Press enter to continue.")

    # part 3: find outliers

    # find a good epsilon threshold using a cross validation set
    Xval = file_data["Xval"]
    yval = file_data["yval"]
    pval = multivariateGaussian(Xval, mu, sigma2)
    epsilon, F1 = selectThreshold(yval, pval)
    print("\nBest epsilon found using cross-validation: {}".format(epsilon))
    print("This should be about 8.99e-5")
    print("\nBest F1 on cross-validation set: {}".format(F1))
    print("This should be about 0.875000")

    # find the outliers in the training set and plot them
    plt.plot(X[(p < epsilon), 0], X[(p < epsilon), 1], 'ro', MarkerSize=10)
    plt.show(block=False)
    c = input("\nProgram paused. Press enter to continue.")

    # part 4: multidimensional outliers

    # load data
    file_data = sio.loadmat("data/ex8data2.mat")
    X = file_data["X"]
    Xval = file_data["Xval"]
    yval = file_data["yval"]

    # apply the same steps to the larger dataset
    mu, sigma2 = estimateGaussian(X)

    # training set
    p = multivariateGaussian(X, mu, sigma2)

    # cross-validation set
    pval = multivariateGaussian(Xval, mu, sigma2)

    # find the best threshold
    epsilon, F1 = selectThreshold(yval, pval)
    print("\nBest epsilon found using cross-validation: {}".format(epsilon))
    print("This should be about 1.38e-18")
    print("\nBest F1 on cross-validation set: {}".format(F1))
    print("This should be about 0.615385")
    print("# outliers found: {}".format(np.sum(p < epsilon)))
    c = input("\nProgram complete. Press enter to exit.")
