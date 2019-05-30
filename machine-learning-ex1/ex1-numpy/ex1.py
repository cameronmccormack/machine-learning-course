import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def warmUpExercise():
    A = np.eye(5)
    print(A)


def computeCost(X, y, theta):
    m = np.size(y)
    J = (1/(2*m)) * np.sum((X @ theta - y)**2)
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    m = np.size(y)
    for i in range(1, num_iters):
        err = (X @ theta) - y
        theta = theta - ((alpha/m) * X.T @ err)
    return theta


## Machine Learning Online Class - Exercise 1: Linear Regression

## Part 1: Basic Function

# Warm-up Exercise
print("Running warmUpExercise ...")
print("5x5 Identity Matrix:")
warmUpExercise()
c = input("Program paused. Press enter to continue.")


## Part 2: Plotting

print("Plotting Data ...")
f = open("data/ex1data1.txt", "r")
X, y = np.loadtxt(f, delimiter=",", unpack=True)
X = np.array([X]).T
y = np.array([y]).T
m = np.size(y)
plt.plot(X, y, 'ro')
plt.show()
c = input("Program paused. Press enter to continue")


## Part 3: Cost and Gradient Descent
X = np.c_[np.ones(m), X]
theta = np.zeros((2, 1))

# gradient descent settings
iterations = 1500
alpha = 0.01

# test cost function
print("Testing the cost function")
J = computeCost(X, y, theta)
print("With theta = [0 ; 0], Cost computed = {}".format(J))
print("Expected cost value (approx) 32.07")

# further testing
J = computeCost(X, y, np.array([[-1], [2]]))
print("\nWith theta = [-1; 2], Cost computed = {}".format(J))
print("Expected cost value (approx) 54.24")
c = input("Program paused. Press enter to continue")

# run gradient descent
print("\nRunning Gradient Descent")
theta = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print("Theta found by gradient descent:")
print(theta)
print("Expected theta values (approx):")
print("-3.6303\n1.1664")

# plot the linear fit
plt.plot(X[:,1], y, 'ro')
plt.plot(X[:,1], X @ theta, color='k', linestyle='-', linewidth=2)
plt.show()
c = input("Program paused. Press enter to continue")

# predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]) @ theta
predict2 = np.array([1, 7]) @ theta
print("For population = 35,000, we predict a profit of {}".format(predict1*10000))
print("For population = 70,000, we predict a profit of {}".format(predict2*10000))
c = input("Program paused. Press enter to continue")


## Part 4: Visualizing J(theta_0, theta_1)
print("Visualizing J(theta_0, theta_1) ...")

# grid over which we calculate J
theta0_vals = np.linspace(-10, 10, num=100)
theta1_vals = np.linspace(-1, 4, num=100)

# initialize J_vals to a matrix of zeros
J_vals = np.zeros((np.size(theta0_vals), np.size(theta1_vals)))

# fill out J_vals
for i in range(0, np.size(theta0_vals)-1):
    for j in range(0, np.size(theta1_vals)-1):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i,j] = computeCost(X, y, t)

