#!/usr/bin/env python3

"""A program that takes DNA sequences as an input from a single external file 
and saves the best alignment along with its score to a new text file"""

__appname__ = '[align_seqs.py]'
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

## imports ## 

import sys
import csv

## functions ##
def get_sequences(path):
    """Read sequences from csv file and append to list"""
    with open(path, 'r') as f:
        csvread = csv.reader(f)
        sequences = []  # To be populated by lines in f
        for row in csvread:
            sequences.append(row)
    return sequences  # Stores as list of list


def get_longer(sequences):
    """Return the longer sequence from input list of sequences"""
    if len(sequences[0][0]) >= len(sequences[1][0]):  # If the length of sequence 0 is greater...
        return sequences[0][0]  # return sequence 0,
    else:
        return sequences[1][0]  # otherwise, return sequence 1


def get_shorter(sequences):
    """Return the shorter sequence from input list of sequences"""
    if get_longer(sequences) == sequences[0][0]:  # Return whichever is not longer
        return sequences[1][0]
    else:
        return sequences[0][0]


def calculate_score(s1, s2, l1, l2, startpoint):
    """Computes score by returning number of matches between sequences s1
    and s2 (where s1 is longer and s2 is shorter, and l1 and l2 are their respective lengths)
    from startpoint of user's choice."""
    matched = ""  # to hold string displaying alignements
    score = 0
    for i in range(l2):
        if (i + startpoint) < l1:  # Only if some part of sequences are aligned, hence seq2[i] + startpoint less than seq1
            if s1[i + startpoint] == s2[i]:  # if the bases match
                matched = matched + "*"  # add asterix to matched
                score = score + 1  # add one to score
            else:
                matched = matched + "-"  # if no match, add dash

    return score


def main(argv):
    """"Loads sequences from file. 
    Finds the best alignment and its score. 
    Saves the results to a text file"""
    my_best_align = None  # set starting values
    my_best_score = -1  # set as >0 to ensure any number of matches is greater

    # Get sequence data
    sequences = get_sequences('../data/Example_seqs.csv')  # Load sequences from csv
    s1 = get_longer(sequences)  # Assign longer to s1
    s2 = get_shorter(sequences)
    l1 = len(s1)
    l2 = len(s2)  # get lengths

    # Find best alignment, update best score
    for i in range(l1):  # Note that you just take the last alignment with the highest score,
        z = calculate_score(s1, s2, l1, l2, i)  # startpoint assigned for each number in longer sequence.
        if z > my_best_score:  # if z is greater than my best score
            my_best_align = "." * i + s2  # reproduces winning alignment where starting point is argument in calculate_score
            my_best_score = z

    # Save results
    rf = "../results/align_seqs.txt"
    with open(rf, 'w') as r:
        r.write(my_best_align + '\n')
        r.write(s1 + '\n')
        r.write("Best score: " + str(my_best_score) + '\n')

    print("Best alignment found!\nResults saved to:", rf)
    return 0


if __name__ == "__main__":
    """If called from console, run main argument"""
    status = main(sys.argv)
    sys.exit(status)
