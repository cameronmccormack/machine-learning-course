import numpy as np
import scipy.io as sio
import imageio
import ex7
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


def projectData(X, U, K):
    U_reduce = U[:, :K]
    Z = X @ U_reduce
    return Z


def recoverData(Z, U, K):
    U_reduce = U[:, :K]
    X_rec = Z @ U_reduce.T
    return X_rec


def displayData(X, example_width=None):
    # set example_width automatically if not passed in
    if not example_width:
        example_width = np.round(np.sqrt(np.size(X, 1))).astype(int)

    # compute rows/columns
    m = np.size(X, 0)
    n = np.size(X, 1)
    example_height = (n / example_width).astype(int)

    # compute number of items to display
    display_rows = np.floor(np.sqrt(m)).astype(int)
    display_cols = np.ceil(m / display_rows).astype(int)

    # set up plot
    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.025, hspace=0.025)
    ax_array = np.atleast_2d(ax_array)
    ax_array = ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_height, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

    plt.show(block=False)


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

    # plot the normalized dataset
    print("\nDimension reduction on example dataset.")
    plt.figure(2)
    plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
    plt.show(block=False)

    # project the data onto K = 1 dimension
    K = 1
    Z = projectData(X_norm, U, K)
    print("\nProjection of the first example: {}".format(Z[0]))
    print("This value should be about 1.481274")
    X_rec = recoverData(Z, U, K)
    print("Approximation of the first example:\n{}\n{}".format(X_rec[0, 0],
                                                               X_rec[0, 1]))
    print("These values should be appoximately:\n-1.047419\n-1.047419")

    # draw lines connecting the projected points to the original points
    plt.figure(3)
    plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
    for i in range(0, np.size(X_norm, 0)):
        plt.plot((X_norm[i, 0], X_rec[i, 0]), (X_norm[i, 1], X_rec[i, 0]),
                 '--k')
    plt.show(block=False)
    c = input("\nProgram paused. Press enter to continue.")

    # part 4: loading and visualizing face data

    # load face dataset
    print("\nLoading face dataset.")
    file_data = sio.loadmat("data/ex7faces.mat")
    X = file_data["X"]
    displayData(X[:100, :])
    c = input("\nProgram paused. Press enter to continue.")

    # part 5: PCA on face data: eigenfaces

    # run PCA and visualize the eigenvectors (in this case eigenfaces)
    print("\nRunning PCA on face dataset.")
    X_norm, mu = featureNormalize(X)
    U, S = pca(X_norm)
    displayData(U[:, :36].T)
    c = input("\nProgram paused. Press enter to continue.")

    # part 6: dimension reduction for faces

    # project images to the eigen space using the top K eigenvectors
    print("Dimension reduction for face dataset.")
    K = 100
    Z = projectData(X_norm, U, K)
    print("The projected data Z as a size of: {}".format(np.size(Z, 0)))
    c = input("\nProgram paused. Press enter to continue.")

    # part 7: visualization of faces after PCA dimension reduction

    # project images to eigen space and visualize using K dimensions
    print("\nVisualizing the projected (reduced dimenesion) faces.")
    K = 100
    X_rec = recoverData(Z, U, K)
    displayData(X_rec[:100, :])
    c = input("\nProgram paused. Press enter to continue.")

    # part 8: PCA for visualization

    # load bird image
    A = imageio.imread("data/bird_small.png")
    A = A/255
    X = np.reshape(A, (np.size(A, 0)*np.size(A, 1), 3))
    K = 16
    max_iters = 10
    initial_centroids = ex7.kMeansInitCentroids(X, K)
    centroids, idx = ex7.runkMeans(X, initial_centroids, max_iters)

    # sample 1000 random indexes (working with all the data is expensive)
    sel = np.floor(np.random.rand(1000, 1) * np.size(X, 0))
