# !/usr/bin/env python3

"""Evolving orbium over ten gradient environments.
In each environment obstacle gradient radiates from single obstacle source and
spans the entire grid (n=1, r=60). Orbium are evolved for 1.5 hours over
10 different gradient strengths (lambda = 0.5:9.5)"""

## IMPORTS ###
from lenia_package import *  # Load package
import re  # regex
import os

# Load creatures

# iter = float(Sys.getenv("PBS_ARRAY_INDEX"))

global g, s
g = 0.5  # set gradient count
s = 0  # set seed

def run_simulations(g, run_time):
    np.random.seed(s)

    orbium = Creature("orbium")  # Initiate creature
    obstacle = ObstacleChannel(n=1, r=60, gradient = g)

    optimise_timely(orbium, obstacle, N=100, run_time = run_time)


# Running simulation commented out
"""for i in range(10):
    run_simulations(i)"""


## EVOLVE FURTHER
run_simulations(3.5, run_time=12)
run_simulations(6.5, run_time=6)

"""RESULTS"""
def sort_data(i):
        """A load of gymnastics and rewriting/saving because I cocked up the
        filenames after 12 hours of simulations"""
        m = re.search(r"mutations\d*", i).group().split("s")
        gradient = re.search(r"g\d\.\d", i).group().split("g")[1]
        theta = []
        with open("parameters/"+i, "r") as f:
            csvread = csv.reader(f)
            for v in csvread:
                theta.append(v)
        theta.append(m)
        theta.append(["gradient", gradient])
        theta = pd.DataFrame(theta)
        theta = theta.transpose()
        theta = np.asarray(theta)

        with open("parameters/gradient_runs2/orbiumt4200n1r60_gradient"+gradient+"_parameters.csv", "w") as f:
            csvwrite = csv.writer(f)
            for row in theta:
                csvwrite.writerow(row)

def sort_times(i):
        """A load of gymnastics and rewriting/saving because I cocked up the
        filenames after 12 hours of simulations"""
        gradient = re.search(r"g\d\.\d", i).group().split("g")[1]
        theta = []
        with open("results/gradient_runs/"+i, "r") as f:
            csvread = csv.reader(f)
            for v in csvread:
                theta.append(v)
        with open("results/gradient_runs2/TIMELOG_orbiumt4200n1r60_gradient"+gradient+".csv", "w") as f:
            csvwrite = csv.writer(f)
            for row in theta:
                csvwrite.writerow(row)

