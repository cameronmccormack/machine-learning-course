import numpy as np
import scipy.io as sio
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
    ax_array = ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_height, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

    plt.show(block=False)


if __name__ == "__main__":
    # set up parameters
    input_layer_size = 400
    num_labels = 10

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
