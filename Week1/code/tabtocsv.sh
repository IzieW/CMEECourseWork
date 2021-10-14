#!/bin/bash
# Author: Izie Wood Iw121@ic.ac.uk
# Script: tabtocsv.sh
# Description: substitute the tabs in the files with commas
#
# Saves the output to .csv file
# Arguments: 1 -> tab delimited file
# Date: Oct 2021

if [ -z $1 ]; then # if no file given, return error message
    echo "Error- Please enter name of file you wish to convert after the command"
    echo "Ex: tabtocsv.sh [FILE]"
    echo "Try again"
    exit
    else 
echo "Creating a comma delimited version of $1..."
cat $1 | tr -s "\t" "," >> $(basename "$1" .txt).csv # tr -s removes tabs, replaces with comma; basename removes .txt from filename, to replace with .csv
echo "Done!"
exit
fi