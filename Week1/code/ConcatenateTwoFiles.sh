#!/bin/bash
# Author: Izie Wood iw121@ic.ac.uk
# Script: ConcatenateTwoFiles.sh
# Desc: Merge two files into new file
# Arguments: 
#   2 -> files to be merged 
#   1 -> file name where merged contents will be saved
# Date: Oct 2021

if [ -z $1 ] | [ -z $2 ] | [ -z $3 ]; then #if file 1, 2, or 3 not given
    echo "Error- Not enough files given"
    echo
    echo "Please give names of the two files you wish to merge, and the name of the file you wish to save the merged contents to, separated by spaces, after the command"
    echo
    echo "Ex: bash ConcatenateTwoFiles.sh FILE1 FILE2 FILE3"
    echo "Try again."
    exit
else    # otherwise...
cat $1 > $3 # Save file1 to file3
cat $2 >> $3  #Append file2 to file3
echo "Merged File is"
cat $3 # Print merged file contents
echo
echo "saved to $3"
exit
fi