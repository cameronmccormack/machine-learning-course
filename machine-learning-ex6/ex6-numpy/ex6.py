import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # part 1: loading and visualizing data

    # load from ex6data1 and plot
    print("Loading and visualizing data ...")
    file_data = sio.loadmat("data/ex6data1.mat")
    X = file_data["X"]
    y = file_data["y"].astype(np.int16)
    plt.figure(1)
    plt.plot(X, y, 'rx')
    plt.show(block=False)
    c = input("\nProgram paused. Press enter to continue")

    # part 2: training linear SVM

    # set C parameter and train
    print("\nTraining linear SVM ...")
    C = 100
    model = svmTrain(X, y, C, "linearKernel", 1e-3, 20)
    visualizeBoundaryLinear(X, y, model)
