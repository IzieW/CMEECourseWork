#!/bin/bash
# Author: Izie Wood Iw121@ic.ac.uk
# Script: tabtocsv.sh
# Description: substitute the tabs in the files with commas
#
# Saves the output to .csv file
# Arguments: 1 -> tab delimited file
# Date: Oct 2021

echo "Creating a comma delimited version of $1..."
cat $1 | tr -s "\t" "," >> $1.csv
echo "Done!"
exit