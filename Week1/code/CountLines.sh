#!/bin/bash
# Author: Izie Wood iw121@ic.ac.uk
# Script: CountLines.sh
# Desc: Count lines in a given file, echo line number
# Arguments: 1 -> file with lines to count
# Date: Oct 2021

# Check for valid files...
if [ -z $1 ]; then # if no file given, return error message 
    echo "Error: no arguments given. Please give the name of the file after the command."
    echo "Ex: CountLines.sh FILENAME"
    exit 2  # Error code 2: insufficient arguments
elif [ ! -f $1 ]; then  #if given file doesn't exist, throw error
    echo "Error in $1: No such file or directory"
    exit 2
fi

# Count lines
NumLines=`wc -l < $1`  # Save result of bash command
echo "The file $1 has $NumLines lines" 
exit 0  # Exit without errors
