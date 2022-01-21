#!/usr/bin/env Python3

"""Imports data from csv, writes file from data containing
only species name and body mass"""

__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

import csv  # Module specialised for csv

# Read a file containing:
# 'Species', 'Infraorder', 'Family', 'Distribution', 'Body mass male (Kg)'
with open('../data/testcsv.csv','r') as f:  # "with" opens and closes files safely
    
    csvread = csv.reader(f)  # iterator to iter over csv 
    temp = []
    print("Reading from ../data/testcsv.csv:\n")
    for row in csvread:
        temp.append(tuple(row))
        print(row)
        print("The species is",row[0])

# write the file containing only species name and Body mass
with open('../data/testcsv.csv','r') as f:
    with open('../results/bodymass.csv','w') as g:

        csvread = csv.reader(f)
        csvwrite = csv.writer(g)
        print("\nWriting only bodymass ../results/bodymass.csv\n")  # feedback to user
        for row in csvread:
            print(row[0], row[4])  # Use only rows for species and bodymass
            csvwrite.writerow([row[0], row[4]])
