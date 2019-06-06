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


def polyFeatures(X, p):
    X_poly = np.zeros((np.size(X), p))
    for i in range(0, p):
        X_poly[:, i] = np.reshape(X**(i+1), np.size(X))
    return X_poly


def featureNormalize(X):
    m = np.size(X, 0)
    n = np.size(X, 1)
    X_norm = np.zeros((m, n))
    for i in range(0, n):
        mu = np.mean(X[:, i])
        sigma = np.std(X[:, i])
        X_norm[:, i] = (X[:, i] - mu) / sigma
    return X_norm, mu, sigma


def validationCurve(X, y, Xval, yval):
    # test these values of lambda
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    # initialize errors
    error_train = np.zeros((np.size(lambda_vec), 1))
    error_val = np.zeros((np.size(lambda_vec), 1))

    for i, lambda_reg in enumerate(lambda_vec):
        print(lambda_reg)
        theta = trainLinearReg(X, y, lambda_reg)
        error_train[i, 0] = linearRegCostFunction(theta, X, y, 0)[0]
        error_val[i, 0] = linearRegCostFunction(theta, Xval, yval, 0)[0]
    return lambda_vec, error_train, error_val


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
    theta = np.atleast_2d(trainLinearReg(X_temp, y, lambda_reg)).T
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
    error_train, error_val = learningCurve(X_temp, y, Xval_temp, yval,
                                           lambda_reg)
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

    # number of features
    p = 8

    # map X onto polynomial features and normalize
    X_poly = polyFeatures(X, p)
    X_poly, mu, sigma = featureNormalize(X_poly)
    X_poly = np.c_[np.ones((m, 1)), X_poly]

    # map X_poly_test and normalize
    Xtest = file_data["Xtest"]
    X_poly_test = polyFeatures(Xtest, p)
    X_poly_test = featureNormalize(X_poly_test)[0]
    X_poly_test = np.c_[np.ones((np.size(X_poly_test, 0), 1)), X_poly_test]

    # map X_poly_val and normalize
    X_poly_val = polyFeatures(Xval, p)
    X_poly_val = featureNormalize(X_poly_val)[0]
    X_poly_val = np.c_[np.ones((np.size(X_poly_val, 0), 1)), X_poly_val]
    print("\nNormalized training example 1:")
    print(X_poly[0, :])
    c = input("\nProgram paused. Press enter to continue.")

    # part 7: learning curve for polynomial regression

    # train with lambda = 0.5
    lambda_reg = 0.5
    theta = np.atleast_2d(trainLinearReg(X_poly, y, lambda_reg)).T

    # plot training data
    plt.figure(3)
    plt.plot(X, y, 'r+')
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.title("Polynomial regression fit (lambda = {})".format(lambda_reg))

    # plot poly fit
    plot_X = np.atleast_2d(np.linspace(np.min(X), np.max(X), 100)).T
    plot_X_poly = polyFeatures(plot_X, p)
    plot_X_poly = featureNormalize(plot_X_poly)[0]
    plot_X_poly = np.c_[np.ones((np.size(plot_X_poly, 0), 1)), plot_X_poly]
    plt.plot(plot_X, plot_X_poly @ theta, '--')
    plt.show(block=False)

    # plot learning curve
    plt.figure(4)
    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval,
                                           lambda_reg)
    plt.plot(range(1, m+1), error_train, label="Train")
    plt.plot(range(1, m+1), error_val, label="Cross Validation")
    plt.title("Learning curve for polynomial linear regression (lambda = {})"
              .format(lambda_reg))
    plt.legend()
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.show(block=False)
    c = input("\nProgram paused. Press enter to continue.")

    # part 8: validation for selecting lambda
    lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val,
                                                         yval)
    plt.figure(5)
    plt.plot(lambda_vec, error_train, label="Train")
    plt.plot(lambda_vec, error_val, label="Cross Validation")
    plt.legend()
    plt.xlabel("Lambda")
    plt.ylabel("Error")
    plt.show(block=False)
    c = input("Program complete. Press enter to exit.")
