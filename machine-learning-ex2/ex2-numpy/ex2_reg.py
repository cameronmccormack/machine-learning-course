import numpy as np
import scipy.optimize as opt
from ex2 import sigmoid, predict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plotData(X, y):
    pos = []
    neg = []
    for i in range(0, np.size(y) - 1):
        if y[i, 0] == 1:
            pos.append(i)
        else:
            neg.append(i)
    plt.plot(X[pos, 0], X[pos, 1], 'k+', label="y=1")
    plt.plot(X[neg, 0], X[neg, 1], 'ko', label="y=0")
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.legend()
    plt.show(block=False)


def plotDecisionBoundary(theta, X, y):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((np.size(u), np.size(v)))
    for i in range(1, np.size(u)):
        for j in range(1, np.size(v)):
            features = mapFeature(np.array([u[i]]), np.array([v[j]]))
            z[i, j] = theta.T @ np.reshape(features, (np.size(features), 1))
    z = z.T
    plt.contour(u, v, z, 0)
    plt.show(block=False)


def mapFeature(X1, X2):
    degree = 6
    out = np.ones((np.size(X1, 0), 1))
    for i in range(1, degree+1):
        for j in range(0, i+1):
            out = np.c_[out, (X1**(i-j))*(X2**j)]
    return out


def costFunctionReg(theta, X, y, reg_lambda):
    m = np.size(y)

    # optimization functions unroll theta, reshape here
    n = np.size(theta)
    theta = np.reshape(theta, (n, 1))

    J = ((1/m) * (-y.T @ np.log(sigmoid(X @ theta))
         - (1 - y.T) @ np.log(1 - sigmoid(X @ theta)))
         + (reg_lambda/(2 * m)) * (np.sum(theta**2) - theta[0, 0]**2))

    grad = np.zeros((np.size(theta), 1))
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y) + (reg_lambda / m) * theta
    grad[0, 0] = grad[0, 0] - (reg_lambda / m) * theta[0, 0]

    return np.asscalar(J), grad


if __name__ == "__main__":

    # Exercise 2 - logistic regression

    # load data
    data = np.loadtxt("data/ex2data2.txt", delimiter=",")
    X, y = data[:, :2], data[:, 2:]

    # plot data

    print("Plotting data with + indicating (y=1) examples and o indicating "
          "(y=0) examples.")
    plotData(X, y)
    c = input("\nProgram paused. Press enter to continue.")

    # part 1: regularized logistic regression

    X = mapFeature(X[:, 0], X[:, 1])
    initial_theta = np.zeros((np.size(X, 1), 1))
    reg_lambda = 1
    print(X[0, :])

    # compute and display initial cost and gradient
    cost, grad = costFunctionReg(initial_theta, X, y, reg_lambda)
    print("\nCost at initial theta (zeros): {}".format(cost))
    print("Expected cost (approx): 0.693")
    print("\nGradient at initial theta (zeros) - first five values only:")
    print(grad[:5, :])
    print("Expected gradients (approx) - first five values only:")
    print("0.0085\n0.0188\n0.0001\n0.0503\n0.0115")
    c = input("\nProgram paused. Press enter to continue.")

    # compute and display cost and gradient with all-ones theta and lambda = 10
    test_theta = np.ones((np.size(X, 1), 1))
    cost, grad = costFunctionReg(test_theta, X, y, 10)
    print("\nCost at test theta with lambda = 10: {}".format(cost))
    print("Expected cost (approx): 3.16")
    print("\nGradient at test theta - first five values only:")
    print(grad[:5, :])
    print("Expected gradients (approx) - first five values only:")
    print("0.3460\n0.1614\n0.1948\n0.2269\n0.0922")
    c = input("\nProgram paused. Press enter to continue.")

    # part 2: regularization and accuracies

    # optimize using parameters from above
    opt_result = opt.fmin_tnc(costFunctionReg, initial_theta,
                              args=(X, y, reg_lambda))
    theta = np.reshape(opt_result[0], (np.size(opt_result[0]), 1))
    cost = costFunctionReg(theta, X, y, reg_lambda)[0]

    # plot boundary
    plotDecisionBoundary(theta, X, y)

    # compute accuracy on our training set
    p = predict(theta, X)
    accuracy = np.mean(np.double(p == y)*100)
    print("Train accuracy: {} %".format(accuracy))
    print("Expected accuracy (approx): 83.1 %")
    c = input("\nProgram complete. Press enter to exit.")
