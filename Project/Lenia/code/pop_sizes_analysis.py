# !/usr/bin/env python3

"""Take input filename from simulation and load vitals:
- Mean survival time and variance
- Parameters
- Survival time evolution"""
import matplotlib.pyplot as plt
import pandas as pd

from lenia_package import *  # Import lenia package
import os
import re

## FUNCTIONS ##
def load_times_pop_size(files):
    files = [i for i in files if re.search(r"times.csv$", i)]
    df = pd.DataFrame( {
        "pop_size": [],
        "wild_mean": [],
        "wild_var": [],
        "mutant_mean": [],
        "mutant_var": []

    })
    for i in files:
        pop_size = int(re.search(r"N\d*", i).group().split("N")[1])  # Separate out pop_size by name
        temp = pd.read_csv("../results/pop_sizes/"+i)
        temp = temp.rename(columns={"Unnamed: 0": "pop_size"})  # Change unnamed column to popsize
        temp.pop_size = pop_size
        df = pd.concat([df, temp])

    return df

def load_vitals(files, path="../results/pop_sizes"):
    files = [i for i in files if re.search(r"parameters.csv$", i)]
    df = pd.DataFrame( {
        "R": [],
        "T": [],
        "m": [],
        "s": [],
        "b": [],
        "mutations": [],
        "gradient": [],
        "survival_mean": [],
        "survival_var": [],
        "pop_size": [],
    })

    for i in files:
        pop_size = int(re.search(r"n\d*", i).group().split("n")[1])  # Separate out pop_size by name
        temp = pd.read_csv(path + i, header=None)
        temp = temp.transpose()
        temp = temp.set_axis(list(temp.iloc[0]), axis = 1)  # rename headers
        temp = temp[1:]
        temp["pop_size"] = pop_size
        df = pd.concat([df, temp])

    return df

def update_survival_times(files):
    files = [i for i in files if re.search(r"parameters.csv$", i)]
    obstacle = ObstacleChannel(n=5, r=8)
    """Compensate for bug in lenia script. Find true survival means and variance by loading, re-running and re-saving survival times"""
    for i in files:
        orbium = Creature("../results/pop_sizes/"+i.split("_parameters.csv")[0], cluster=True)
        times = get_survival_time(orbium, obstacle, summary=True)
        orbium.survival_mean = times[0]
        orbium.survival_var = times[1]
        with open(orbium.name+"_parameters.csv", "w") as f:
            csvwrite = csv.writer(f)
            for k in Creature.keys:
                csvwrite.writerow([k, orbium.__dict__[k]])
            csvwrite.writerow(["mutations", orbium.mutations])
            csvwrite.writerow(["gradient", orbium.evolved_in])
            csvwrite.writerow(["survival_mean", orbium.survival_mean])
            csvwrite.writerow(["survival_var", orbium.survival_var])





#### PLOT DATA ####
def plot_theta(*orbium, variable=None):
    """Display any number of theta on single axis
    plt.figure(figsize=(5, 2.7), layout = "constrained")
    for i in range(len(orbium)):
        plt.bar(["Radius", "Time", "Growth Mean", "Growth std", "kernel peaks"], [orbium[i].R, orbium[i].T, orbium[i].m, orbium[i].s, orbium[i].b], label = variable[i])

    plt.xlabel("Parameter")
    plt.ylabel("value")
    plt.legend()"""
    x = np.linspace(0, 1, 1000)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(len(orbium)):
        ax1.bar(["Radius", "Time", "kernel peaks"], [orbium[i].R, orbium[i].T, orbium[i].b], label = variable[i])
        ax2.plot(x, orbium[i].growth(U=x), label = variable[i])
        ax2.title.set_text("Growth Function")
    plt.xlabel("Parameters")
    plt.ylabel("values")
    plt.legend()

######## MULTIPLE ORBIUM #########
def orbium_counter(seed):
    """Return number of orbiums in the run based on seed number.
    This ammendment is due to a fuck up on my part"""
    count = np.repeat([1, 2, 3, 4, 5, 10], 3)
    seeds = np.arange(0, 17)
    counter = {}
    for i in range(17):
        counter[seeds[i]] = count[i]

    return counter[seed]




