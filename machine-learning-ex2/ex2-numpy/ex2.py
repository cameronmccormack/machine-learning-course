import numpy as np
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


if __name__ == "__main__":

    # Exercise 2 - logistic regression

    # load data
    data = np.loadtxt("data/ex2data1.txt", delimiter=",")
    X, y = data[:, :2], data[:, 2:]

    print(X)
    print(y)

    # part 1: plotting

    print("Plotting data with + indicating (y=1) examples and o indicating "
          "(y=0) examples.")
    plotData(X, y)
    c = input("Program paused. Press enter to continue.")

    # part 2: compute cost and gradient
