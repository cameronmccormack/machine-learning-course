import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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


def sigmoid(z):
    denom = 1 + np.exp(-z)
    return denom**-1


def lrCostFunction(theta, X, y, reg_lambda):
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


def oneVsAll(X, y, num_labels, lambda_reg):
    # add ones to X data matrix
    m = np.size(X, 0)
    X = np.c_[np.ones((m, 1)), X]

    # more useful variables
    n = np.size(X, 1)
    all_theta = np.zeros((num_labels, n))

    # train logistic regression classifiers
    for i in range(0, num_labels):
        initial_theta = np.zeros((n, 1))
        y_temp = np.where(y == i+1, 1, 0)
        opt_result = opt.fmin_tnc(lrCostFunction, x0=initial_theta,
                                  args=(X, y_temp, lambda_reg))
        all_theta[i, :] = opt_result[0]
    return all_theta


def predictOneVsAll(all_theta, X):
    m = np.size(X, 0)
    p = np.zeros((m, 1))
    X = np.c_[np.ones((m, 1)), X]
    p = np.argmax(X @ all_theta.T, 1) + 1
    return np.reshape(p, (m, 1))


if __name__ == "__main__":
    # set up parameters
    input_layer_size = 400
    num_labels = 10

    # part 1: loading and visualizing data

    # load training data
    print("Loading and visualizing data ...")
    file_data = sio.loadmat("data/ex3data1.mat")
    X = file_data["X"]
    y = file_data["y"]
    m = np.size(X, 0)

    # randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]
    displayData(sel)
    c = input("\nProgram paused. Press enter to continue.")

    # part 2a: vectorize logistic regression

    # test case for lrCostFunction
    print("\nTesting lrCostFunction with regularization")
    theta_t = np.array([[-2], [-1], [1], [2]])
    X_t = np.c_[np.ones((5, 1)), np.reshape(range(1, 16), (3, 5)).T/10]
    y_t = np.array([[1], [0], [1], [0], [1]])
    lambda_t = 3
    J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

    print("\nCost: {}".format(J))
    print("Expected cost: 2.534819")
    print("\nGradients:")
    print(grad)
    print("Expected gradients:")
    print("0.146561\n-0.548558\n0.724722\n1.398003")
    c = input("\nProgram paused. Press enter to continue.")

    # part 2b: one-vs-all training

    print("\nTraining one-vs-all logistic regression ...")
    lambda_reg = 0.1
    all_theta = oneVsAll(X, y, num_labels, lambda_reg)
    c = input("\nProgram paused. Press enter to continue.")

    # part 3: predict for one-vs-all

    pred = predictOneVsAll(all_theta, X)
    accuracy = np.mean(np.double(pred == y)*100)
    print("\n Training set accuracy: {}".format(accuracy))
    c = input("\nProgram complete. Press enter to exit.")
