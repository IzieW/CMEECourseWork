#!/usr/bin/bash
# Author: Izie Wood iw121@ic.ac.uk
# Script: csvtospace.sh 
# Desc: Takes comma separated values and converts to space separates values, saves to new file
# Argument: 1 -> comma delimited file
# Date: Oct 2021

# Check for valid input files...
if [ -z $1 ]; then # if no file given, return error message and exit
    echo "Error: filename missing. Please give the file name you wish to convert after the command"
    echo "Ex: bash csvtospaces.sh FILENAME"
    exit 2  # Insufficient arguments
elif [ ! -f $1 ]; then # if file nonexistent, throw error
    echo "Error in $1: no such file or directory"
    exit 2
fi

# Create space separated file from input
    echo "Creating a space delimited version of $1 ..."
    cat $1 | tr -s "," " " >> ../results/$(basename -s .csv $1)_spaces.txt  #  Replace commas with spaces, save as txt file in results
    echo "Done! file saved to ../results/$(basename -s .csv $1)_spaces.txt"

exit 0  #Exit successfully