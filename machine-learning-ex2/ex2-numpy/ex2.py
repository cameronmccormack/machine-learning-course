import numpy as np
import scipy.optimize as opt
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

    plt.plot(X[pos, 0], X[pos, 1], 'k+', label="Admitted")
    plt.plot(X[neg, 0], X[neg, 1], 'ko', label="Not admitted")
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.legend()
    plt.show(block=False)


def sigmoid(z):
    denom = 1 + np.exp(-z)
    return denom**-1


def costFunction(theta, X, y):
    m = np.size(y)
    n = np.size(theta)
    theta = np.reshape(theta, (n, 1))
    J = (1/m) * (-y.T @ np.log(sigmoid(X @ theta))
                 - (1-y).T @ np.log(1 - sigmoid(X @ theta)))
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y)
    return np.asscalar(J), grad


if __name__ == "__main__":

    # Exercise 2 - logistic regression

    # load data
    data = np.loadtxt("data/ex2data1.txt", delimiter=",")
    X, y = data[:, :2], data[:, 2:]

    # part 1: plotting

    print("Plotting data with + indicating (y=1) examples and o indicating "
          "(y=0) examples.")
    plotData(X, y)
    c = input("\nProgram paused. Press enter to continue.")

    # part 2: compute cost and gradient

    # add ones for intercept term
    m = np.size(X, 0)
    n = np.size(X, 1)
    X = np.c_[np.ones((m, 1)), X]

    # initialize fitting parameters
    initial_theta = np.zeros(n+1)

    # compute and display initial cost and gradient
    cost, grad = costFunction(initial_theta, X, y)
    print("\nCost at initial theta (zeros): {}".format(cost))
    print("Expected cost (approx): 0.693")
    print("\nGradient at initial theta (zeros):")
    print(grad)
    print("Expected gradient (approx):\n-0.1000\n-12.0092\n-11.2628")

    # compute and display cost and gradient with non-zero theta
    test_theta = np.array([[-24], [0.2], [0.2]])
    cost, grad = costFunction(test_theta, X, y)
    print("\nCost at test theta: {}".format(cost))
    print("Expected cost (approx): 0.218")
    print("\nGradient at thest theta:")
    print(grad)
    print("Expected gradient (approx):\n0.043\n2.566\n2.647")
    c = input("\nProgram paused. Press enter to continue.")

    # part 3: optimizing
    opt_result = opt.fmin_tnc(costFunction, initial_theta, args=(X,y))
    theta = np.reshape(opt_result[0], (3,1))
    cost = costFunction(theta, X, y)[0]

    # print cost and theta
    print("\nCost at theta found by fmin_tnc: {}".format(cost))
    print("Expected cost (approx): 0.203")
    print("\ntheta:")
    print(theta)
    print("Expected theta (approx):\n-25.161\n0.206\n0.201")

    # plot boundary
    plotDecisionBoundary(theta, X, y)
    c = input("\nProgram paused. Press enter to continue.")
