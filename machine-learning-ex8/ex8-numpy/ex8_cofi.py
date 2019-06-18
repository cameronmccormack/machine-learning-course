import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features,
                 lambda_reg):
    # unfold X and Theta
    X = np.reshape(params[:(num_movies*num_features), 0],
                   (num_movies, num_features))
    Theta = np.reshape(params[(num_movies*num_features):, 0],
                       (num_users, num_features))

    # compute cost function and gradient
    M = ((Theta @ X.T).T - Y) * R
    J = 0.5 * np.sum(M**2)
    X_grad = M @ Theta
    Theta_grad = M.T @ X

    # add regularization
    J = J + (lambda_reg/2) * (np.sum(Theta**2) + np.sum(X**2))
    X_grad = X_grad + lambda_reg * X
    Theta_grad = Theta_grad + lambda_reg * Theta

    # refold grads
    grad = np.r_[np.reshape(X_grad, (np.size(X_grad), 1)),
                 np.reshape(Theta_grad, (np.size(Theta_grad), 1))]
    return J, grad


def checkCostFunction(lambda_reg=0):
    # gradient checking to be added
    pass


if __name__ == "__main__":
    # part 1: loading movie ratings dataset

    # load data
    print("Loading movie ratings dataset.")
    file_data = sio.loadmat("data/ex8_movies.mat")
    Y = file_data["Y"]
    R = file_data["R"]

    # compute statistics
    print("Average rating for movie 1 (Toy Story): {}"
          .format(np.mean(Y[R == 1])))

    # visualize ratings
    plt.figure()
    plt.imshow(Y)
    plt.colorbar()
    plt.ylabel("Movies")
    plt.xlabel("Users")
    plt.show(block=False)
    c = input("\nProgram paused. Press enter to continue.")

    # part 2: collaborative filtering cost function

    # load pre-trained weights
    file_data = sio.loadmat("data/ex8_movieParams.mat")
    X = file_data["X"]
    Theta = file_data["Theta"]
    num_users = file_data["num_users"]
    num_movies = file_data["num_movies"]
    num_features = file_data["num_features"]

    # reduce the data set size so this runs faster
    num_users, num_movies, num_features = 4, 5, 3
    X = X[:num_movies, :num_features]
    Theta = Theta[:num_users, :num_features]
    Y = Y[:num_movies, :num_users]
    R = R[:num_movies, :num_users]

    # evaluate cost function
    cofi_input = np.r_[np.reshape(X, (np.size(X), 1)),
                       np.reshape(Theta, (np.size(Theta), 1))]
    J, grad = cofiCostFunc(cofi_input, Y, R, num_users, num_movies,
                           num_features, 0)
    print("\nCost at loaded parameters: {}".format(J))
    print("This value should be about 22.22")
    c = input("\nProgram paused. Press enter to continue.")

    # part 3: collaborative filtering gradient

    # check gradient function
    print("\nChecking gradients (without regularization).")
    checkCostFunction
    c = input("\nProgram paused. Press enter to continue.")

    # part 4: collaborative filtering cost regularization

    # evaluate cost function with lambda = 1.5
    lambda_reg = 1.5
    J, grad = cofiCostFunc(cofi_input, Y, R, num_users, num_movies,
                           num_features, lambda_reg)

    print("\nCost at loaded parameters: {}".format(J))
    print("This value should be about 31.34")
    c = input("\nProgram paused. Press enter to continue.")

    # part 5: collaborative filtering gradient regularization

    # check gradient with regularization
    print("\nChecking gradients (with regularization).")
    checkCostFunction(lambda_reg)
    c = input("\nProgram paused. Press enter to continue.")

    # part 6: entering rating for a new user

    # load movies
    
