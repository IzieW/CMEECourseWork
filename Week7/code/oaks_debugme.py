#!/usr/bin/env Python3 
""" Optimised oaks_debugme.py script using doctest """
__author__ = '{Izie Wood (iw121@ic.ac.uk)'

############# imports ###########
import csv
import sys
import doctest
from difflib import SequenceMatcher


#Define function
def oak_similarity(a):
    """Calculate similarity between given string and 'quercus'
    >>> oak_similarity('Quercus')
    1.0
    >>> oak_similarity('Fagus')
    0.3333333333333333
    >>> oak_similarity('Quercuss')
    0.9333333333333333
    """
    return SequenceMatcher(None, a.lower(), "quercus").ratio() #return similarity of two strings in ratio of similarity

def main(argv): 
    f = open('../data/TestOaksData.csv','r')
    g = open('../data/JustOaksData.csv','w')
    taxa = csv.reader(f)
    csvwrite = csv.writer(g)
    oaks = set()
    for row in taxa:
        print(row)
        print ("The genus is: ")
        print(row[0] + '\n')
        if oak_similarity(row[0]) == 1.0:
            print("FOUND AN OAK!")
            csvwrite.writerow([row[0], row[1]])
        elif oak_similarity(row[0]) >= 0.8:
            print("Possible oak detected- Check for spelling")

    return 0
    
if (__name__ == "__main__"):
    status = main(sys.argv)

doctest.testmod()