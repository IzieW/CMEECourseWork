# Author: Izie Wood (iw121@ic.ac.uk)
# Script: basic_csv.py
# Desc: scripts demonstrating reading and writing csv
# Arguments: 2 -> csv files
# Date: Oct 2021

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