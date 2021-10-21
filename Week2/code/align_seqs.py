#!/usr/bin/env python3

"""A program that takes DNA sequences as an input from a single external file 
and saves the best alignment along with its score in a new text file"""

__appname__ = '[align_seqs.py]'
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

## imports ## 

import csv # package to manipulate csv

import sys

## functions ##
# Read sequences from csv file and append to list 
with open('../data/Example_seqs.csv', 'r') as f:
    
    csvread = csv.reader(f) # Iterate over lines in f
    sequences = [] # To be populated by lines in f
    for row in csvread: 
        sequences.append(row)
    
# Assign sequences to variables
# Assign the longer sequence s1, and the shorter to s2
# l1 is length of the longest, l2 that of the shortest
seq1 = sequences[0][0]
seq2 = sequences[1][0]

l1 = len(seq1)
l2 = len(seq2)
if l1 >= l2:
    s1 = seq1
    s2 = seq2
else:
    s1 = seq2
    s2 = seq1
    l1, l2 = l2, l1 # swap the two lengths

# A function that computes a score by returning the number of matches starting
# from arbitrary startpoint (chosen by user)
def calculate_score(s1, s2, l1, l2, startpoint):
    matched = "" # to hold string displaying alignements
    score = 0
    for i in range(l2):
        if (i + startpoint) < l1: # Only if some part of sequences are aligned, hence seq2[i] + startpoint less than seq1 
            if s1[i + startpoint] == s2[i]: # if the bases match
                matched = matched + "*" # add asterix to matched
                score = score + 1 # add one to score
            else:
                matched = matched + "-" # if no match, add dash

    return score

# Test the function with some example starting points:
# calculate_score(s1, s2, l1, l2, 0)
# calculate_score(s1, s2, l1, l2, 1)
# calculate_score(s1, s2, l1, l2, 5)

# now try to find the best match (highest score) for the two sequences
def main(argv):
    """main entry point of argument"""    
    my_best_align = None
    my_best_score = -1

    for i in range(l1): # Note that you just take the last alignment with the highest score,
        z = calculate_score(s1, s2, l1, l2, i) # startpoint assigned for each number in longer sequence.
        if z > my_best_score: # if z is greater than my best score
            my_best_align = "." * i + s2 # reproduces winning alignment where starting point is argument in calculate_score
            my_best_score = z 

# Saves best alignment and score to txt file 
    rf = "../results/align_seqs.txt"
    with open(rf, 'w') as r:
        r.write(my_best_align + '\n')
        r.write(s1 + '\n')
        r.write("Best score: "+ str(my_best_score) + '\n')
    
    print("Best alignment found!\nResults saved to:",rf)
    return 0 

if __name__ == "__main__":
    status = main(sys.argv)
    sys.exit(status)
