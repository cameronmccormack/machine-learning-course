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


def sigmoidGradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels,
                   X, y, lambda_reg):
    # setup parameters
    nn_params = np.reshape(np.atleast_2d(nn_params), (np.size(nn_params), 1))
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1),
                                  :],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):,
                                  :],
                        (num_labels, hidden_layer_size + 1))
    m = np.size(X, 0)
    Theta1_grad = np.zeros((np.size(Theta1, 0), np.size(Theta1, 1)))
    Theta2_grad = np.zeros((np.size(Theta2, 0), np.size(Theta2, 1)))

    # forward propogation
    a1 = np.c_[np.ones((m, 1)), X]
    z2 = a1 @ Theta1.T
    a2 = np.c_[np.ones((m, 1)), sigmoid(z2)]
    z3 = a2 @ Theta2.T
    a3 = sigmoid(z3)

    # vectorise y
    Y = np.zeros((m, num_labels))
    for i in range(0, m):
        Y[i, y[i, 0] - 1] = 1

    # compute cost
    J = (1/m) * np.sum(-Y * np.log(a3) - (1 - Y) * np.log(1 - a3))

    # add regularization to cost
    J = J + (lambda_reg/(2*m))*(np.sum(Theta1_temp ** 2)
                                + np.sum(Theta2_temp ** 2)
                                - np.sum(Theta1[:, 0]**2)
                                - np.sum(Theta2[:, 0]**2))

    # back propogate with regularization
    d3 = a3 - Y
    d2_temp = d3 @ Theta2
    d2_temp = np.delete(d2_temp, (0), axis=1)
    d2 = d2_temp * sigmoidGradient(z2)
    Theta2_grad = Theta2_grad + (1/m) * (d3.T @ a2) + (lambda_reg/m) * Theta2
    Theta1_grad = Theta1_grad + (1/m) * (d2.T @ a1) + (lambda_reg/m) * Theta1
    Theta2_grad[:, 0] = Theta2_grad[:, 0] - (lambda_reg/m) * Theta2[:, 0]
    Theta1_grad[:, 0] = Theta1_grad[:, 0] - (lambda_reg/m) * Theta1[:, 0]

    Theta1_grad = np.resize(Theta1_grad, (np.size(Theta1_grad), 1))
    Theta2_grad = np.resize(Theta2_grad, (np.size(Theta2_grad), 1))
    grad = np.r_[Theta1_grad, Theta2_grad]

    return J, grad


def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W

def Predict(Theta1, Theta2, X):
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
    # setup parameters
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    # part 1: loading and visualizing data

    # load training data
    print("Loading and visualizing data ...")
    file_data = sio.loadmat("data/ex4data1.mat")
    X = file_data["X"]
    y = file_data["y"]
    m = np.size(X, 0)

    # randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]
    displayData(sel)
    c = input("\nProgram paused. Press enter to continue.")

    # part 2: loading parameters

    # load weights
    print("\nLoading saved neural network parameters ...")
    file_data = sio.loadmat("data/ex4weights.mat")
    Theta1 = file_data["Theta1"]
    Theta2 = file_data["Theta2"]

    # unroll parameters
    Theta1_temp = np.resize(Theta1, (np.size(Theta1), 1))
    Theta2_temp = np.resize(Theta2, (np.size(Theta2), 1))
    nn_params = np.r_[Theta1_temp, Theta2_temp]

    # part 3: compute cost (feedforward)

    # set regularization parameter to zero
    print("\nFeedforward using neural network ...")
    lambda_reg = 0
    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                       num_labels, X, y, lambda_reg)[0]
    print("\nCost at parameters (loaded from ex4weights): {}".format(J))
    print("This value should be about 0.287629")
    c = input("\nProgram paused. Press enter to continue.")

    # part 4: implement regularization

    # set regularization parameter to 1 and regularize
    print("\nChecking cost function (with regularization) ...")
    lambda_reg = 1
    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                       num_labels, X, y, lambda_reg)[0]
    print("\nCost at parameters (loaded from ex4weights): {}".format(J))
    print("This value should be about 0.383770")
    c = input("\nProgram paused. Press enter to continue.")

    # part 5: sigmoid gradient

    # test sigmoid gradient function
    print("\nEvaluating sigmoid gradient ...")
    test = np.array([-1, -0.5, 0, 0.5, 1])
    g = sigmoidGradient(test)
    print("\nSigmoid gradient evaluated at {}:\n{}".format(test, g))
    c = input("\nProgram paused. Press enter to continue.")

    # part 6: initializing parameters

    # randomly initialize parameter weights close to zero
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    # unroll
    initial_Theta1 = np.resize(initial_Theta1, (np.size(initial_Theta1), 1))
    initial_Theta2 = np.resize(initial_Theta2, (np.size(initial_Theta2), 1))
    initial_nn_params = np.r_[initial_Theta1, initial_Theta2]

    # part 7: implement backpropagation and regularization

    # backpropogation and regularization are implemented in nnCostFunction

    # part 8: training nn

    # optimize theta values
    print("\nTraining neural network")
    lambda_reg = 1
    opt_result = opt.fmin_tnc(nnCostFunction, x0=initial_nn_params,
                              args=(input_layer_size, hidden_layer_size,
                                    num_labels, X, y, lambda_reg),
                              maxfun=200)
    nn_params = opt_result[0]

    # reshape nn_params to obtain theta1 and theta2
    nn_params = np.reshape(np.atleast_2d(nn_params), (np.size(nn_params), 1))
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1),
                                  :],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):,
                                  :],
                        (num_labels, hidden_layer_size + 1))
    c = input("\nProgram paused. Press enter to continue.")

    # part 9: visualize weights

    print("\nVisualizing neural network ...")
    displayData(Theta1[:, 1:])
    c = input("\nProgram paused. Press enter to continue.")

    # part 10: implement predict

    pred = Predict(Theta1, Theta2, X)
    accuracy = np.mean(np.double(pred == y)*100)
    print("\nTraining set accuracy: {}".format(accuracy))
    c = input("\nProgram complete. Press enter to exit")
