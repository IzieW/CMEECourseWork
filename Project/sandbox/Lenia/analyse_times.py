# !/usr/bin/env python3

"""Analyse timelog solutions from fixation runs"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

colorscheme = ["darkslategrey", "lightslategray", "slateblue"]
def load_data(filename):
    return pd.read_csv("results/time_logs/"+filename+"_times.csv")

dat = load_data(filename)

runs = np.arange(len(dat))


plt.plot(runs, dat["wild"], color="darkslategrey")
plt.xlabel("selection step")
plt.ylabel("survival time")
plt.suptitle("Orbium survival time over selection process")
plt.title("fixation 20, seed 0", fontsize = 10)
plt.vlines(33, ymin=0, ymax = 10000, colors=colorscheme[1], linestyles="dashed", linewidth=1)
plt.annotate("fixation_10", xy=(23, 4000), xytext=(23, 4000), fontsize=7)
