#!/usr/bin/bash
# Author: Izie Wood iw121@ic.ac.uk
# Script: ConcatenateTwoFiles.sh
# Desc: Merge two input files into new file
# Arguments: 
#   2 -> files to be merged 
#   1 -> (optional) file name where merged contents will be saved 
# Date: Oct 2021

# Check valid files given to merge
if [ -z $1 ] | [ -z $2 ]; then # If two files to merge not given
    echo "Error: Missing Arguments. Please give the two files to be merged after the command"
    echo "Ex: bash ConcatenateTwoFiles.sh FILE1 FILE2 [DESTINATION]"
    exit 2
elif [ ! -f $1 ]; then # If either file is nonexistent, throw error and exit
    echo "Error in $1: No such file or directory"
    exit 2
elif [ ! -f $2 ]; then
    echo "Error in $2: No such file or directory"
    exit 2
fi

# Check if destination is specificed in argument...
if [ -z $3 ]; then # If name of merged file not specificied,  use concatenation of two given file names
    OUTPUT_FILE=../results/$(basename ${2::-4})_$(basename $1) # remove path from filenames, take suffix from file 1
else 
    OUTPUT_FILE=$(echo $3)  # Otherwise use argument given at command line
fi

# Merge files to new file
cat $1 > $OUTPUT_FILE # Save file1 to output
cat $2 >> $OUTPUT_FILE  #Append file2 to file3

echo "Merged File is:"
cat $OUTPUT_FILE # Print merged file contents
echo
echo "saved to $OUTPUT_FILE"  # Tell user where results are saved
exit 0 # Code ran successfully
