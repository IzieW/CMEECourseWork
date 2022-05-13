# !/usr/bin/env python3

"""Creates plot of survival times over evolutionary runs to overview optimisation process"""

## IMPORTS ##
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

colorscheme = ["darkslategrey", "lightslategray", "slateblue"]

def load_data(filename):
    return pd.read_csv("results/time_logs/" + filename + "_times.csv")


def main(argv):
    """Return time plot from filename"""
    filename = argv[1]
    subtitle = argv[2]
    dat = load_data(filename)
    runs = np.arange(len(dat))
    plt.plot(runs, dat["wild"], color="darkslategrey")
    plt.xlabel("selection step")
    plt.ylabel("survival time")
    plt.suptitle("Orbium survival time over selection process")
    plt.title(subtitle, fontsize=10)
    #plt.vlines(33, ymin=0, ymax=10000, colors="lightslategray", linestyles="dashed", linewidth=1)
    #plt.annotate("fixation_10", xy=(23, 4000), xytext=(23, 4000), fontsize=7)
    plt.matshow()

if __name__ == "__main__":
    main(sys.argv)
