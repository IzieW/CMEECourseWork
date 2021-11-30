########################################## Mini Project ########################################
##################################### Work flow 1: Data Prep ###################################
#!/usr/bin/env python3
"""Miniproject Work flow 1/3:
 Loads and prepares data from LogisticGrowthData.csv for model fitting.
        - Creates unique ID to distinguish between unique datasets 
        - Deals with missing or problematic values
        - Saves modified data to CSV file """

__author__ = "Izie Wood (iw121@ic.ac.uk)"

######### Imports #########
import pandas as pd
import scipy as sc
import matplotlib.pylab as py
import seaborn as sns 

######## Functions ########
## Load data from csv
data = pd.read_csv("../data/LogisticGrowthData.csv") # load and save data
print("Loaded {} columns.".format(len(data.columns.values))) # count columns
print(data.columns.values) # column names 

## Create unique ID's for each individuals growth curves
data.insert(0,"ID", data.Species + "_" + data.Temp.map(str) + "_" + data.Medium + "_" + 
data.Citation)  
