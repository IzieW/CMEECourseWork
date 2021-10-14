#!/bin/bash
# Author: Izie Wood iw121@ic.ac.uk
# Script: CountLines.sh
# Desc: Count lines in a given file, echo line number
# Arguments: 1 -> file with lines to count
# Date: Oct 2021

if [ -z $1 ]; then # if no file given, return error message 
    echo "Error- Please give the name of the file after the command"
    echo "ex: CountLines.sh FILE"
    echo "Try again"
else
NumLines=`wc -l < $1`
echo "The file $1 has $NumLines lines"
fi
exit