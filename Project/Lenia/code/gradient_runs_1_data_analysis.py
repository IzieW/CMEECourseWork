# !/usr/bin/env python3

"""Evolving orbium over ten gradient environments.
In each environment obstacle gradient radiates from single obstacle source and
spans the entire grid (n=1, r=60). Orbium are evolved for 1.5 hours over
10 different gradient strengths (lambda = 0.5:9.5)"""

## IMPORTS ###
from lenia_package import *  # Load package
import re  # regex

"""RESULTS: A  bunch of gymnastics to retrieve results from last gradient run"""
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


