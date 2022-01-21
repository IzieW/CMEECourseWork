#!/usr/bin/env Python3 
""" Debugged doc-tested script that identifies oaks in a list of trees, and can 
handle potential typos """
__author__ = '{Izie Wood (iw121@ic.ac.uk)'

############# imports ###########
import csv
import sys
import doctest
from difflib import SequenceMatcher  # Module for matching sequences of strings


def oak_similarity(a):
    """Calculate similarity between given string and 'quercus'
    >>> oak_similarity('Quercus')
    1.0
    >>> oak_similarity('Fagus')
    0.3333333333333333
    >>> oak_similarity('Quercuss')
    0.9333333333333333
    >>> oak_similarity('')
    0.0
    """
    return SequenceMatcher(None, a.lower(), "quercus").ratio()  #  Return similarity of two strings in ratio of similarity

def main(argv):
    """Print oak data and flag when oak is found"""
    with open('../data/TestOaksData.csv','r') as f, open('../results/JustOaksData.csv','w') as g:
        taxa = csv.reader(f)
        csvwrite = csv.writer(g)
        oaks = set()
        for row in taxa:
            print(row)
            print("The genus is: ")
            print(row[0] + '\n')
            if oak_similarity(row[0]) == 1.0:  # If similarity is a perfect match... flag oak and save to list
                print("FOUND AN OAK!")
                csvwrite.writerow([row[0], row[1]])
            elif oak_similarity(row[0]) >= 0.8:  # If strings are sufficiently similar, flag potential typo
                print("Possible oak detected- Check for spelling")
    print("Done! All oaks saved to ../results/JustOaksData.csv")
    return 0


if __name__ == "__main__":
    """When called from command line, run main function"""
    status = main(sys.argv)

doctest.testmod()
