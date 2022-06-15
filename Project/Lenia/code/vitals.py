# !/usr/bin/env python3
"""Script to read in and organise data from optimisation runs"""

from lenia_package import *
import re
import os

###### FUNCTIONS ######
def get_files(path):
    """Pulls list of files from input directory in results directory"""
    return os.listdir(path)

def get_seed(file):
    """Return seed from input filename"""
    return int(re.search(r"s\d*", file).group().split("s")[1])

def get_popsize(file):
    """Return seed from input filename"""
    return int(re.search(r"n\d*", file).group().split("n")[1])

def load_vitals(path):
    """Load in parameter information from all files. Return in one dataframe"""
    files = get_files(path)
    files = [i for i in files if re.search(r"parameters.csv$", i)]
    pop_sizes = [get_popsize(i) for i in files]
    seeds = [get_seed(i) for i in files]
    # Kick off data frame with first value in list
    df = pd.read_csv(path+files[0], header=None)
    df = df.transpose()
    df = df.set_axis(list(df.iloc[0]), axis = 1)  # Tranpose and rename header with column names
    df = df[1:]
    df["pop_size"] = pop_sizes[0]
    df["seed"] = seeds[0]

    for i in range(1, len(files)):
        temp = pd.read_csv(path+files[i], header=None)
        temp = temp.transpose()
        temp = temp.set_axis(list(temp.iloc[0]), axis = 1)  # rename headers
        temp = temp[1:]
        temp["pop_size"] = pop_sizes[i]
        temp["seed"] = seeds[i]

        df = pd.concat([df, temp])

    return df.astype("int64")

