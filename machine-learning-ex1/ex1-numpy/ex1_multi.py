import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import ex1


def featureNormalize(X):
    features = np.size(X, 1)
    X_norm = X
    mu = np.zeros((1, features))
    sigma = np.zeros((1, features))
    for i in range(0, features):
        mu[0, i] = np.mean(X[:, i])
        sigma[0, i] = np.std(X[:, i])
        X_norm[:, i] = (X[:, i] - mu[0, i])/sigma[0, i]
    return X_norm, mu, sigma


def normalEqn(X, y):
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return theta


if __name__ == "__main__":
    # Exercise 1 - Linear regression with multiple variables

    # Part 1: Feature Normalization

    # load data
    print("Loading data...")
    data = np.loadtxt("data/ex1data2.txt", delimiter=",")
    X, y = data[:, :2], data[:, 2:]
    m = np.size(y)

    # print out some data points
    print("First 10 examples from the dataset:")
    for i in range(0, 9):
        print("x = {}, y = {}".format(X[i, :], y[i, :]))
    c = input("Program paused. Press enter to continue.")

    # scale features and set them to zero mean
    print("Normalizing features...")
    X, mu, sigma = featureNormalize(X)

    # add intercept term to X
    X = np.c_[np.ones((m, 1)), X]

    # Part 2: Gradient descent

    # set parameters
    print("Running gradient descent...")
    alpha = 0.1
    num_iters = 400

    # init theta and run gradient descent
    theta = np.zeros((3, 1))
    theta, J_history = ex1.gradientDescent(X, y, theta, alpha, num_iters)

    # plot the convergence graph
    plt.plot(np.linspace(1, np.size(J_history), np.size(J_history)),
             J_history, '-b')
    plt.show(block=False)

    # display gradient descent's result
    print("Theta computed from gradient descent:")
    print(theta)

    # estimate the price of a 1650 sq-ft, 3 br house
    x = (np.array([1, 1650, 3]) - np.c_[[0], mu]) / np.c_[[1], sigma]
    x = x.T
    price = theta.T @ x
    print("Predicted price of a 1650 sq-ft, 3 br house using gradient "
          "descent: ${}".format(np.asscalar(price)))
    c = input("Program paused. Press enter to continue.")

    # Part 3: normal equations

    # reload data
    data = np.loadtxt("data/ex1data2.txt", delimiter=",")
    X, y = data[:, :2], data[:, 2:]
    X = np.c_[np.ones((m, 1)), X]

    # calculate the parameters from the normal equation
    theta = normalEqn(X, y)

    # display normal equation's result
    print("Theta computed from the normal equations:")
    print(theta)

    # estimate the price of a 1650 sq-ft, 3 br house
    x = np.array([[1], [1650], [3]])
    price = theta.T @ x
    print("Predicted price of a 1650 sq-ft, 3 br house using normal "
          "equations: ${}".format(np.asscalar(price)))
