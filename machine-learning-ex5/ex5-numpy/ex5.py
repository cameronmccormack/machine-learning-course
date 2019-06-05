import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def linearRegCostFunction(theta, X, y, lambda_reg):
    m = np.size(np.atleast_2d(y), 0)
    theta = np.reshape(np.atleast_2d(theta), (np.size(theta), 1))
    theta_temp = theta.copy()
    theta_temp[0, 0] = 0
    J = ((1/(2*m))*np.sum((X @ theta - y)**2)
         + (lambda_reg/(2*m))*np.sum(theta_temp))
    grad = (1/m) * (X.T @ (X @ theta - y)) + (lambda_reg/m) * theta_temp
    return J, grad


def trainLinearReg(X, y, lambda_reg):
    initial_theta = np.zeros((np.size(X, 1), 1))
    opt_result = opt.fmin_tnc(linearRegCostFunction, initial_theta,
                              args=(X, y, lambda_reg), maxfun=200)
    theta = opt_result[0]
    return theta


def learningCurve(X, y, Xval, yval, lambda_reg):
    # number of training examples
    m = np.size(X, 0)

    # prepare error vectors
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    # calculate errors
    for i in range(0, m):
        theta = trainLinearReg(X[:i+1, :], y[:i+1, :], lambda_reg)
        theta = np.reshape(np.atleast_2d(theta), (np.size(theta), 1))
        error_train[i, 0] = linearRegCostFunction(theta, X[:i+1, :],
                                                  y[:i+1, :], 0)[0]
        error_val[i, 0] = linearRegCostFunction(theta, Xval, yval, 0)[0]
    return error_train, error_val


if __name__ == "__main__":

    # part 1: loading and visualizing data

    # load from ex5data1
    print("Loading and visualizing data ...")
    file_data = sio.loadmat("data/ex5data1.mat")
    X = file_data["X"]
    y = file_data["y"]
    m = np.size(X, 0)

    # plot training data
    plt.figure(1)
    plt.plot(X, y, 'rx')
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.show(block=False)
    c = input("\nProgram paused. Press enter to continue.")

    # part 2: regularized linear regression cost

    # test cost function
    theta = np.array([[1], [1]])
    X_temp = np.c_[np.ones((m, 1)), X]
    J = linearRegCostFunction(theta, X_temp, y, 1)[0]
    print("\nCost at theta = [1; 1]: {}".format(J))
    print("This value should be about 303.993192")
    c = input("\nProgram paused. Press enter to continue.")

    # part 3: regularized linear regression gradient

    # test gradient
    theta = np.array([[1], [1]])
    X_temp = np.c_[np.ones((m, 1)), X]
    grad = linearRegCostFunction(theta, X_temp, y, 1)[1]
    print("\nGradient at theta = [1; 1]:")
    print(grad)
    print("This gradient should be about:\n-15.303016\n598.250744")
    c = input("\nProgram paused. Press enter to continue.")

    # part 4: train linear regression

    # train linear regression with lambda_reg = 0
    lambda_reg = 0
    theta = trainLinearReg(X_temp, y, lambda_reg)
    plt.plot(X, y, 'rx')
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.plot(X, X_temp @ theta, '--')
    plt.show(block=False)
    c = input("\nProgram paused. Press enter to continue.")

    # part 5: learning curve for linear regression

    # import validation set
    Xval = file_data["Xval"]
    Xval_temp = np.c_[np.ones((np.size(Xval, 0), 1)), Xval]
    yval = file_data["yval"]

    # plot learning curve
    lambda_reg = 0
    error_train, error_val = learningCurve(X_temp, y, Xval_temp, yval, lambda_reg)
    plt.figure(2)
    plt.plot(range(1, m+1), error_train, label="Train")
    plt.plot(range(1, m+1), error_val, label="Cross Validation")
    plt.title("Learning curve for linear regression")
    plt.legend()
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.show(block=False)
    c = input("\nProgram paused. Press enter to continue.")

    # part 6: feature mapping for polynomial regression
