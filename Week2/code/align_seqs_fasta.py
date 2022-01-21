#!/usr/bin/env python3
"""Python program that aligns two DNA sequences and calculates the best
alignment score, given some input files.
"""
__author__ = 'Dashing Dingos'
__version__ = '0.0.1'

## imports ##
import sys

## functions ##
def calculate_score(
    s1: str, s2: str, startpoint: int = 0, verbose: bool = False
):
    """ A function computing a score based on alignment.

    This computes a score by returning the number of matches starting
    from arbitrary offset (chosen by user) of the first sequence against the
    second, denoted by the startpoint.
    """
    if startpoint >= len(s2):
        raise ValueError("Offset provided higher than second sequence length")

    matched = "" # to hold string displaying alignments
    score = 0
    for i in range(startpoint, min(len(s1)+startpoint, len(s2))):
        # Iterate over the minimum length, to avoid steps which can't do
        # any comparison.
        if s2[i] == s1[i-startpoint]: # if the bases match
            matched += "*"
            score += 1
        else:
            matched += "-"

    # some formatted output
    aligned = "." * startpoint + s1
    if verbose:
        print("." * startpoint + matched)
        print(aligned)
        print(s2)
        print(score)
        print(" ")

    return score, aligned


def best_match(shortest: str, longest: str, verbose: bool = False):
    """
    A function to align the sequences and calculate all the matches with the
    highest score between the two.

    Returns a triple of a maximum score, the corresponding offset, and a string
    representation of the alignment.
    """
    my_best_score: int = -1
    my_best_offset: Optional[int] = None
    my_best_align: Optional[str] = None

    # append instances of the best score to a list
    for i in range(len(longest)):
        score, aligned = calculate_score(shortest, longest, i, verbose)
        if score > my_best_score:
            my_best_score = score
            my_best_offset = i
            my_best_align = aligned
    return my_best_score, my_best_offset, my_best_align


def read_fasta(file1: str, file2: str):
    """
    A function that reads in 2 fasta files and outputs strings
    of the contents.
    """
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        # Skip the first line, since this is a header.
        next(f1)
        next(f2)
        seq1 = str(''.join(f1.read().splitlines()))
        seq2 = str(''.join(f2.read().splitlines()))
    
    return seq1, seq2


def main(argv, verbose=False):
    """Do the analysis, and save output."""
    argc = len(argv)
    if argc == 1:
        # Default to two fasta files if none provided
        file1 = "../data/fasta/407228326.fasta"
        file2 = "../data/fasta/407228412.fasta"
    elif argc == 3:
        file1 = argv[1]
        file2 = argv[2]
    else:
        print("Usage: align_seqs_better.py [FILE1 FILE2]")
        return 1

    seq1, seq2 = read_fasta(file1, file2)
    try:
        shortest, longest = sorted((seq1, seq2), key=len)
        best_score, best_offset, best_align = best_match(
            shortest, longest, verbose
        )
    except ValueError:
        return 1

    print("Best score:", best_score)

    output_file = '../results/align_fasta.txt'
    with open(output_file, 'w') as f:
        # Produce a new output file with best alignment score from your
        # two input files
        f.write('\n'.join([
            'Align: ' + best_align,
            'Score: ' + str(best_score),
            'Offset: ' + str(best_offset)
        ]) + '\n')

    print("Best alignment found!\nResults saved to " + output_file)
    return 0


if __name__ == "__main__":
    """Run main function when called"""
    status = main(sys.argv)
    sys.exit(status)

