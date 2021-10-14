#!/bin/bash
# Author: Izie Wood iw121@ic.ac.uk
# Script: tiff2png.sh
# Desc: Convert tif files to png
# Arguments: 1 -> tif file 
# Date: Oct 2021

for f in *.tif; # Tif files must be in working directory
    do
        if [ -e $f ]; then # Check tif file exists in working directory
            echo "Converting $f";
            convert "$f" "$(basename "$f" .tif).png";
            echo "Done!"
        else # if not tif files
        echo "No tif file found in directory"
        fi
        exit
    done 