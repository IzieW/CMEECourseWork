#!/usr/bin/env python3

"""A script that finds the heights of provided tree data.

Outputs to target file, or the results directory with a helpful name if not
provided.

Usage: ./get_TreeHeight.py input [output]
"""

import csv
import math
import os.path
import sys
import warnings


def tree_height(degrees, distance):
    """This function calculates heights of trees given distance of each tree
    from its base and angle to its top, using the trigonometric formula.

    height = distance * tan(radians)

    Args:
        degrees:  the angle of elevation to the top of the tree
        distance: the distance from base of tree (e.g., meters)

    Returns:
        The heights of the tree, same units as "distance"
    """
    radians = math.radians(degrees)
    height = distance * math.tan(radians)
    print("Tree height is:", height)
    return height


def main(argv):
    """Read the tree data, and find heights. Write the result."""
    # Handle inputs. Output file is optional.
    out = None
    argc = len(argv)
    if argc == 1:
        # If no input file provided, use a default
        inp = "../data/trees.csv"
    elif argc == 2:
        inp = argv[1]
    elif argc == 3:
        inp = argv[1]
        out = argv[2]
    else:
        warnings.warn('Usage: ./get_TreeHeight.py input [output]')
        return 1

    if not out:
        # Extract input filename, remove file extension and use it in
        # the output filename, if not provided.
        base = os.path.basename(inp).split('.')[0]  # strip .csv
        out = '../results/' + base + '_treeheights_py.csv'

    print("Calculating Tree height...")
    with open(inp,'r') as f, open(out, 'w') as g:
        # Create input iterator and remove the header
        csvread = csv.reader(f)
        header = next(csvread)

        # Write output (with header)
        csvwrite = csv.writer(g)
        header.append("Height.m")
        csvwrite.writerow(header)

        # CSV reader and writer work top down to process rows, therefore we
        # must:
        # - iterate every row in one loop
        # - read the values we need from each row to compute the height
        # - calculate the computed height
        # - write the computed height with the original row
        for row in csvread:
            distance = float(row[1])
            angle = float(row[2])
            row.append(tree_height(angle, distance))
            csvwrite.writerow(row)

    print("Complete! File saved to", out)
    return 0


if __name__ == "__main__":
    """Make sure main function is called from command line"""
    status = main(sys.argv)
    sys.exit(status)

