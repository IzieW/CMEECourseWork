# !/usr/bin/env python3

"""
Run and render lenia with any input set of parameters.

Pass name of file from command line. File should be in the results/parameters folder. No additional path or .csv
is required. Only the bare file name- which is in the results/parameters folder.

Lenia will run parameters and produce two files:
    1. A kernel cross section + growth function png, named after the input filenmae
    2. A rendered animation of the life form from input parameters
"""
#### PREPARATION ####
## IMPORTS ##
import numpy as np
import scipy as sc
from scipy.signal import convolve2d
import matplotlib.pylab as plt
from matplotlib import animation
import sys
import csv

# Silence warnings
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Silence warnings


# Visualisation functions
def figure_world(A, cmap="viridis"):
    """Set up basic graphics of unpopulated, unsized world"""
    global img  # make final image global
    fig = plt.figure()  # Initiate figure
    img = plt.imshow(A, cmap=cmap, interpolation="nearest", vmin=0)  # Set image
    plt.title = ("World A")
    plt.close()
    return fig


def figure_asset(K, growth, cmap="viridis", K_sum=1, bar_K=False):
    """ Return graphical representation of input kernel and growth function.
    Subplot 1: Graph of Kernel in matrix form
    Subplot 2: Cross section of Kernel around center. Y: gives values of cell in row, X: gives column number
    Subplot 3: Growth function according to values of U (Y: growth value, X: values in U)
    """
    global R
    K_size = K.shape[0];
    K_mid = K_size // 2  # Get size and middle of Kernel
    fig, ax = plt.subplots(1, 3, figsize=(14, 2),
                           gridspec_kw={"width_ratios": [1, 1, 2]})  # Initiate figures with subplots

    ax[0].imshow(K, cmap=cmap, interpolation="nearest", vmin=0)
    ax[0].title.set_text("Kernel_K")

    if bar_K:
        ax[1].bar(range(K_size), K[K_mid, :], width=1)  # make bar plot
    else:
        ax[1].plot(range(K_size), K[K_mid, :])  # otherwise, plot normally
    ax[1].title.set_text("K cross-section")
    ax[1].set_xlim([K_mid - R - 3, K_mid + R + 3])

    if K_sum <= 1:
        x = np.linspace(0, K_sum, 1000)
        ax[2].plot(x, growth(x))
    else:
        x = np.arange(K_sum + 1)
        ax[2].step(x, growth(x))
    ax[2].axhline(y=0, color="grey", linestyle="dotted")
    ax[2].title.set_text("Growth G")
    return fig


##### RUN AND RENDER LENIA #####
def load_parameters(filename):
    """Load parameters from csv"""
    dict = {}
    with open("results/parameters/parameters_" + filename + ".csv", "r") as f:
        csvread = csv.reader(f)
        for row in csvread:
            if row[0] == "b":
                dict[row[0]] = [float(i) for i in row[1].strip("[]").split(",")]
            else:
                dict[row[0]] = float(row[1])
    cells = []
    with open("results/parameters/cells_" + filename + ".csv", "r") as f:
        csvread = csv.reader(f)
        for i in csvread:
            cells.append([float(s) for s in i])
    dict["cells"] = cells
    return dict

def load_obstacles(n, size, r=5, seed = 0):
    """Load obstacle channel with random configuration
    of n obstacles with radius r"""
    # Sample center point coordinates a, b
    np.random.seed(seed)
    O = np.zeros([size, size])
    for i in range(n):
        mid_point = tuple(np.random.randint(0, size - 1, 2))
        O[mid_point[0]:mid_point[0] + r, mid_point[1]:mid_point[1] + r] = 1  # load obstacles
    return O

bell = lambda x, m, s: np.exp(-((x - m) / s) ** 2 / 2)  # Gaussian function

def growth(U):
    return bell(U, m, s) * 2 - 1


def obstacle_growth(U):
    return -10 * np.maximum(0, (U - 0.001))


def update(i):
    global As, img
    U1 = np.real(np.fft.ifft2(fK*np.fft.fft2(As[0])))
    #U1 = convolve2d(As[0], K, mode="same", boundary="wrap")
    """Update learning channel with growth from both obstacle and 
    growth channel"""
    As[0] = np.clip(As[0] + 1 / T * (growth(U1) + obstacle_growth(As[1])), 0, 1)
    img.set_array(sum(As))  # Sum two channels to create one channel
    return img,


def main(argv):
    """Compile and render LEnia animation from input parameters"""
    filename = argv[1]
    orbium = load_parameters(filename)

    size = 64;
    mid = size // 2;
    cx, cy = 20, 20
    globals().update(orbium)  # load orbium pattern

    """Load learning channel"""
    C = np.asarray(cells)
    A = np.zeros([size, size])  # Initialise learning channel, A
    A[cx:cx + C.shape[0], cy:cy + C.shape[1]] = C  # Load initial configurations into learning channel)

    """Load obstacle configuration"""
    O = load_obstacles(n=6, size = size)

    """Create kernel for learning channel"""
    D = np.linalg.norm(np.ogrid[-mid:mid, -mid:mid])/R
    K = (D < 1) * bell(D, 0.5, 0.15)  ## Transform all distances within radius 1 along smooth gaussian gradient
    K = K / np.sum(K)  # Normalise between 0:1

    global As, fK  # Make properties global
    fK = np.fft.fft2(np.fft.fftshift(K))  # fourier transform kernel
    As = [A, O]  # List of channels

    figure_asset(K, growth)
    plt.savefig("results/"+filename+"_kernel.png")

    print("rendering animation...")
    fig = figure_world(sum(As))
    anim = animation.FuncAnimation(fig, update, frames=200, interval=20)
    anim.save("results/"+filename+"_anim.gif", writer="imagemagick")
    print("process complete")


if __name__ == "__main__":
    main(sys.argv)
