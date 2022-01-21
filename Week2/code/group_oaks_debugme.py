#! /usr/bin/env python3

"""This is a program that opens a csv file containing species data, 
iterates through the rows and prints "This is an oak!" when 'Quercus' is
identified. It then writes a new csv file containing all of the found Quercus
individuals.
"""
__author__ = 'DashingDingos'

import csv
import sys

def is_an_oak(name: str):
    """Indicates if the species is in the genus "Quercus".

    Accounts for typos up to a Levenshtein distance of two, after accounting
    for case.
    >>> is_an_oak('Quercus robur')
    True
    >>> is_an_oak('Quercus  robur')
    True
    >>> is_an_oak('quercus robur')
    True
    >>> is_an_oak('Quercuss robur')
    False
    >>> is_an_oak('Quecrus robur')
    False
    >>> is_an_oak('Pinus sylvestris')
    False
    >>> is_an_oak('')
    False
    """
    if not name:
        return False
    genus = name.split()[0]
    return genus.lower() == 'quercus'


def main(argv):
    """Read the file, calculate, and produce output."""
    with open('../data/TestOaksData.csv','r') as f, \
         open('../results/JustOaksData.csv','w') as g:
        # Write output
        csvwrite = csv.writer(g)

        # Create input iterator and copy the header to the new file
        taxa = csv.reader(f)
        csvwrite.writerow(next(taxa))

        # Iterate and process rows
        for row in taxa:
            print(row)
            print ("The genus is:")
            print(row[0] + '\n')
            if is_an_oak(' '.join(row)):
                print('FOUND AN OAK!\n')
                csvwrite.writerow(row)
    return 0


if __name__ == "__main__":
    status = main(sys.argv)
    sys.exit(status)

