# !/usr/bin/env python3
"""Script to read in results from 10hr evolution of orbium
across 20 gradient environments.

Orbium were optimised for 10 hours in each environment, each environment running on three seeds.

Script reads in and organises results by environmental gradient"""

## Imports ##
import pandas as pd

from lenia_package import *
import os
import re

## Load data ##
path = lambda x: "../results/gradient_runs_10_hour/"+x

files = os.listdir("../results/gradient_runs_10_hour")
par = [i for i in files if re.search("_parameters*", i)]  # Get list of all parameter files
times = [i for i in files if re.search("_times*", i)]  # Get list of all time files

all_gradients = np.unique([float(re.search(r"\d\.\d", i).group()) for i in par])[1:]  # remove gradient -


def get_data_types():
    keys = Creature.keys
    keys.append("mutations"); keys.append("gradient"); keys.append("survival_mean"); keys.append("survival_var")
    dict = {}
    n = "float64"
    for i in keys:
        dict[i] = n

    dict["b"] = "object"

    return dict

def make_df(filename):
    """Take file name and load to pandas data frame of desired style"""
    df = pd.read_csv(path(filename), header=None)
    df = df.T
    df.columns = df.iloc[0]
    return df[1:]

def group_by_gradient(gradient):
    """Takes input gradient value and finds all files with that gradient.
    Saves to data frame"""
    grad_files = [i for i in par if re.search(r"\d\.\d", i).group() == str(gradient)]  # Get all files with gradient
    seeds = [re.search(r"s..", i).group().split("s")[1] for i in grad_files]  # take all seeds from file
    df = make_df(grad_files[0])
    for i in range(1, len(grad_files)):
        temp = make_df(grad_files[i])
        df = pd.concat([df, temp])

    df["seed"] = seeds  # Add seeds
    df.index = np.arange(len(df.R))  # Add row indices

    df = df.astype(get_data_types())  # Set data types
    return df

def load_all():
    """Load all files from directory into one large dataframe"""
    df = group_by_gradient(all_gradients[0])
    for i in range(1, len(all_gradients)):
        temp = group_by_gradient(all_gradients[i])
        df = pd.concat([df, temp])

    return df

def load_max():
    """Return data frame with parameter values of maximum survival times
    for each gradient run"""
    df = group_by_gradient(all_gradients[0])
    df = df[df.survival_mean == df.survival_mean.max()][0:1] # get row with maximum survival mean

    for i in range(1, len(all_gradients)):
        temp = group_by_gradient(all_gradients[i])
        temp = temp[temp.survival_mean == temp.survival_mean.max()][0:1]
        df = pd.concat([df, temp])

    df.index = np.arange(len(df.R))

    return df


### Analysis   ####
dfm = load_max()
plt.plot(dfm.gradient, np.log(dfm.survival_mean))
plt.show()
