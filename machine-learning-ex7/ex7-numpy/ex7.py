import numpy as np
import scipy.io as sio
import imageio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def findClosestCentroids(X, centroids):
    # set m
    m = np.size(X, 0)

    # initialize idx
    idx = np.zeros((np.size(X, 0), 1))

    # find closest centroids
    for i in range(0, m):
        costs = np.sum((centroids - X[i, :])**2, 1)
        index = np.argmin(costs)
        idx[i, 0] = index
    return idx


def computeCentroids(X, idx, K):
    # useful variables
    m = np.size(X, 0)
    n = np.size(X, 1)

    # initialize centroids
    centroids = np.zeros((K, n))

    # compute mean of points assigned to each centroids
    for i in range(0, K):
        x_store = []
        for j in range(0, m):
            if idx[j, 0] == i:
                x_store.append(X[j, :])
        centroids[i, :] = np.mean(x_store, axis=0)
    return centroids


def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    # initialize values
    m = np.size(X, 0)
    K = np.size(initial_centroids, 0)
    centroids = initial_centroids
    idx = np.zeros((m, 1))
    previous_centroids = initial_centroids

    # run k means
    for i in range(0, max_iters):
        print("\nK-Means iteration {}/{} ...".format(i+1, max_iters))

        # for each example, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)

        # compute new centroids
        centroids = computeCentroids(X, idx, K)

        # optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            input("Press enter to continue.")

    return centroids, idx


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    # plot the examples
    plotDataPoints(X, idx, K)

    # plot centroids as black x's
    plt.plot(centroids[:, 0], centroids[:, 1], 'kx', markersize=10)
    plt.show(block=False)
    for i in range(0, np.size(centroids, 0)):
        plt.plot((centroids[i, 0], previous[i, 0]),
                 (centroids[i, 1], previous[i, 1]), 'r-')
        plt.show(block=False)


def plotDataPoints(X, idx, K):
    plt.plot(X[:, 0], X[:, 1], 'bx')
    plt.show(block=False)


def kMeansInitCentroids(X, K):
    random_idx = np.random.permutation(np.size(X, 0))
    centroids = X[random_idx[:K], :]
    return centroids


if __name__ == "__main__":
    # part 1: find closest centroids

    # load data
    print("Finding closest centroids ...")
    file_data = sio.loadmat("data/ex7data2.mat")
    X = file_data["X"]

    # set initial set of centroids and find closest
    K = 3
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    idx = findClosestCentroids(X, initial_centroids)
    print("\nClosest centroids for the first 3 examples:")
    print(idx[0:3, :])
    print("The closest centroids should be 0, 2, 1 respectively")
    c = input("\nProgram paused. Press enter to continue.")

    # part 2: compute means

    print("\nComputing centroids' means")
    centroids = computeCentroids(X, idx, K)
    print("\nCentroids computed after initial finding of closest centroids:")
    print(centroids)
    print("The centroids should be:")
    print("[2.428301, 3.157924]\n[5.813503, 2.633656]\n[7.119387, 3.616684]")
    c = input("\nProgram paused. Press enter to continue.")

    # part 3: K-means clustering

    # settings for running K-means
    max_iters = 10

    # for consistency, use specific initial centroids from before rather than
    # random
    plt.figure(1)
    centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
    print("\nK-Means Done.")
    c = input("\nProgram paused. Press enter to continue.")

    # part 4: K-means clustering on pixels

    # load image
    print("\nRunning K-Means clustering on pixels from an image.")
    A = imageio.imread("data/ski_pic.png")
    A = A/255
    X = np.reshape(A, (np.size(A, 0)*np.size(A, 1), 3))

    # run K-means on this data
    K = 30
    max_iters = 10
    initial_centroids = kMeansInitCentroids(X, K)
    centroids, idx = runkMeans(X, initial_centroids, max_iters)
    c = input("\nProgram paused. Press enter to continue.")

    # part 5: image compression

    print("\nApplying K-Means to compress an image.")

    # find closest cluster members
    idx = findClosestCentroids(X, centroids).astype(int)
    X_recovered = centroids[idx, :]
    X_recovered = np.reshape(X_recovered, (np.size(A, 0), np.size(A, 1), 3))

    # display original and compressed image
    plt.figure(2)
    plt.imshow(A)
    plt.figure(3)
    plt.imshow(X_recovered)
    plt.show(block=False)
    c = input("\nProgram complete. Press enter to exit.")
