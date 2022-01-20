#!/usr/bin/bash
# Author: Izie Wood Iw121@ic.ac.uk
# Script: tabtocsv.sh
# Description: Take tab delimmited file from command line and convert to csv
# Saves the output to .csv file
#
# Arguments: 1 -> tab delimited file
# Date: Oct 2021

# Check valid file given
if [ -z $1 ]; then # if no file given, return error message
    echo "Error: no arguments given. Please enter name of file you wish to convert after the command."
    echo "Ex: bash tabtocsv.sh FILENAME"
    exit 2  # Exit code: arguments missing
elif [ ! -f $1 ]; then  # If given file doesn't exist, throw error
    echo "Error in $1: No such file or directory"
    exit 2 
fi

# Convert to csv
echo "Creating a comma delimited version of $1 ..."
# Create name of output file
FILENAME=../results/$( basename -s .txt $1).csv  # basename to removes path, -s argument removes .txt suffix, new path to results directory
cat $1 | tr -s "\t" "," >> $FILENAME #  Replaces tabs with commas, save to output file
echo "Done! File saved to" $FILENAME  # Tell user where results are saved
exit 0  # Code ran successfully

