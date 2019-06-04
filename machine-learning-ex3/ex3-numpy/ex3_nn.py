import numpy as np
import scipy.io as sio
import scipy.optimize as opt
from ex3 import displayData, sigmoid
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def predict(Theta1, Theta2, X):
    # useful variables
    m = np.size(X, 0)

    # initialize prediction
    p = np.zeros((m, 1))

    # add bias terms to X data
    X = np.c_[np.ones((m, 1)), X]

    # predictions using neural network
    a1 = X
    z2 = a1 @ Theta1.T
    a2 = np.c_[np.ones((m, 1)), sigmoid(z2)]
    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)
    p = np.argmax(a3, 1) + 1
    return np.reshape(p, (m, 1))


if __name__ == "__main__":
    # set up parameters
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    # part 1: loading and visualizing data

    # load training data
    print("Loading and Visualizing Data ...")
    file_data = sio.loadmat("data/ex3data1.mat")
    X = file_data["X"]
    y = file_data["y"]
    m = np.size(X, 0)

    # randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]
    displayData(sel)
    c = input("\nProgram paused. Press enter to continue.")

    # part 2: loading parameters

    print("\nLoading saved neural network parameters ...")
    file_data = sio.loadmat("data/ex3weights.mat")
    Theta1 = file_data["Theta1"]
    Theta2 = file_data["Theta2"]

    # part 3: implement predict

    # predictions for all data
    pred = predict(Theta1, Theta2, X)
    accuracy = np.mean(np.double(pred == y)*100)
    print("\nTraining set accuracy: {}".format(accuracy))
    c = input("\nProgram paused. Press enter to continue.")

    # prediction for data one at a time
    rp = np.random.permutation(m)

    for i in range(0, m):
        # display image
        print("\nDisplaying example image.")
        X_one = np.atleast_2d(X[rp[i], :])
        displayData(X_one)

        # predict number
        pred = predict(Theta1, Theta2, X_one)
        print("Neural network prediction: {}".format(pred))

        # pause with quit option
        c = input("\nPaused - press enter to continue, "
                  "type q then enter to exit. ")
        if c == "q" or c == "Q":
            break
