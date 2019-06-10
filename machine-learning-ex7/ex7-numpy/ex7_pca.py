import numpy as np
import scipy.io as sio
import imageio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def featureNormalize(X):
    mu = np.mean(X, axis=0)
    X_norm = (X - mu)/np.std(X, axis=0)
    return X_norm, mu


def pca(X):
    m = np.size(X, 0)
    Sigma = (1/m) * X.T @ X
    U, S, V = np.linalg.svd(Sigma)
    return U, S


if __name__ == "__main__":
    # part 1: load example dataset

    # visualize
    print("Visualizing example dataset for PCA.")
    file_data = sio.loadmat("data/ex7data1.mat")
    X = file_data["X"]
    plt.figure(1)
    plt.plot(X[:, 0], X[:, 1], 'bo')
    plt.show(block=False)
    c = input("\nProgram paused. Press enter to continue.")

    # part 2: principal component analysis

    # normalize X and run PCA
    print("\nRunning PCA on example dataset.")
    X_norm, mu = featureNormalize(X)
    U, S = pca(X_norm)

    # draw the eigenvactors centered at mean of data
    plt.plot((mu[0], mu[0] + S[0] * U[0, 0]),
             (mu[1], mu[1] + S[0] * U[1, 0]), 'r-')
    plt.show(block=False)
    print("Top eigenvector:")
    print("{}\n{}".format(U[0, 0], U[1, 0]))
    print("Expected values:\n-0.707107\n-0.707107")
    c = input("\nProgram paused. Press enter to continue.")

    # part 3: dimension reduction


