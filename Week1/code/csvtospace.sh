#!/bin/bash
# Author: Izie Wood iw121@ic.ac.uk
# Script: csvtospace.sh 
# Desc: Takes comma separated values and converts to space separates values
# Argument: 1 -> comma delimited file
# Date: Oct 2021

if [ -z $1 ]; then # if no file given to convert, return error message and exit
    echo "Error- Please give the file name you wish to convert after the command"
    echo "Ex: csvtospaces.sh FILE"
    echo "Try again."
else # otherwise, proceed with conversion to space separated file
    echo "Creating a space delimited version of $1..."
    cat $1 | tr -s "," " " >> $1_spaces.txt
    echo "Done!"
fi
exit