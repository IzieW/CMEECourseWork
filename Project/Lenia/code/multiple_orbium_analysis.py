# !/usr/bin/env python3

"""Load data from quick run with multiple orbium.
Find vitals to scan for interesting patterns"""
import matplotlib.pyplot as plt
import pandas as pd

from lenia_package import *  # Import lenia package
import os
import re


##### FUNCTIONS #####
def get_seed(file):
    """Return seed from input filename"""
    return int(re.search(r"s\d*", file).group().split("s")[1])


def orbium_counter(seed):
    """Return number of orbiums in the run based on seed number.
    This ammendment is due to a fuck up on my part"""
    count = np.repeat([1, 2, 3, 4, 5, 10], 3)
    seeds = np.arange(0, 18)
    counter = {}
    for i in range(18):
        counter[seeds[i]] = count[i]

    return counter[seed]


def load_vitals():
    files = os.listdir("../results/multiple_orbium")
    files = [i for i in files if re.search(r"parameters.csv$", i)]  # find parameter files only
    df = pd.DataFrame({
        "R": [],
        "T": [],
        "m": [],
        "s": [],
        "b": [],
        "mutations": [],
        "gradient": [],
        "survival_mean": [],
        "survival_var": [],
        "organism_count": [],
    })


def get_survival_means(files):
    """Function to ammend fuck up in survival means.
    Load orbium from input files and recalculate survival mean and var."""
    ## first need to get number of orbium in each file based on the seed
    files = [i for i in files if re.search(r"parameters.csv$", i)]  # find parameter files only
    seeds = [get_seed(i) for i in files]
    N = [orbium_counter(i) for i in seeds]  # get orbium count
    obstacle = ObstacleChannel(n=5, r=8)
    for i in range(len(files)):
        orbium = Creature("../results/multiple_orbium/" + files[i].split("_parameters.csv")[0],
                          n=N[i], cluster=True)
        times = get_survival_time(orbium, obstacle, summary=True)
        orbium.survival_mean = times[0]
        orbium.survival_var = times[1]

        orbium.name = orbium.name + "_parameters.csv"

        with open(orbium.name, "w") as f:
            csvwrite = csv.writer(f)
            for k in Creature.keys:
                csvwrite.writerow([k, orbium.__dict__[k]])
            csvwrite.writerow(["mutations", orbium.mutations])
            csvwrite.writerow(["gradient", orbium.evolved_in])
            csvwrite.writerow(["survival_mean", orbium.survival_mean])
            csvwrite.writerow(["survival_var", orbium.survival_var])
            csvwrite.writerow(["organism_count", orbium.n])


def re_write_files(file):
    """Function to load and re-write faulty files...
    again to correct my own fuck up"""
    temp = []
    with open("../results/multiple_orbium/" + file, "r") as f:
        csvread = csv.reader(f)
        for row in csvread:
            temp.append(row[1])
    theta = [temp[0], temp[6], temp[12], temp[18], temp[24]]
    rest = [temp[i] for i in range(2, 7)]
    names = ["mutations", "gradient", "survival_mean", "survival_var"]

    with open("../results/multiple_orbium/" + file, "w") as f:
        csvwrite = csv.writer(f)
        for i in range(len(theta)):
            csvwrite.writerow([Creature.keys[i], theta[i]])
        for i in range(len(names)):
            csvwrite.writerow([names[i], rest[i]])
