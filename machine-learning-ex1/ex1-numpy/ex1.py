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


## Machine Learning Online Class - Exercise 1: Linear Regression

## Part 1: Basic Function

# Warm-up Exercise
print("Running warmUpExercise ...")
print("5x5 Identity Matrix:")
warmUpExercise()
c = input("Program paused. Press enter to continue.")


## Part 2: Plotting

print("Plotting Data ...")
f = open("ex1data1.txt", "r")
X, y = np.loadtxt(f, delimiter=",", unpack=True)
m = np.size(y)
plt.plot(X, y, 'ro')
#plt.show()
c = input("Program paused. Press enter to continue.")


## Part 3: Cost and Gradient Descent

X = np.append([np.ones(m)], [X], axis=0)
X = X.transpose()
y = y.transpose()
theta = np.zeros((2, 1))

# gradient descent settings
iterations = 1500
alpha = 0.01

print("Testing the cost function")
J = computeCost(X, y, theta)
print("With theta = [0 ; 0], Cost computed = {}".format(J))

