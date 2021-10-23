#!/usr/bin/env Python3

"""Script illustrating how to read and write files from csv"""
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

import csv

# Read a file containing:
# 'Species', 'Infraorder', 'Family', 'Distribution', 'Body mass male (Kg)'
with open('../data/testcsv.csv','r') as f:

    csvread = csv.reader(f)
    temp = []
    for row in csvread:
        temp.append(tuple(row))
        print(row)
        print("The species is",row[0])

# write the file containing only species name and Body mass
with open('../data/testcsv.csv','r') as f:
    with open('../data/bodymass.csv','w') as g:

        csvread = csv.reader(f)
        csvwrite = csv.writer(g)
        for row in csvread:
            print(row)
            csvwrite.writerow([row[0], row[4]])