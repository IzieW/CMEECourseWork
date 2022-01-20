#!/usr/bin/bash
# Author: Izie Wood iw121@ic.ac.uk
# Script: tiff2png.sh
# Desc: Converts all tiff files in working directory to png
# Arguments: 1 -> None
# Date: Oct 2021
# Dependencies: imagemagick

for f in *.tif; # for every .tif file in the working directory...
    do
        if [ -e $f ]; then # If the file exists, convert to png
            echo "Converting $f";
            convert "$f" "$(basename "$f" .tif).png";
            echo "Done!"
        else # if no .tif files, throw warning and exit
        echo "Warning: No tiff files in directory"
        fi
    done
        exit 0  # Both finish successfully since task is to convert all .tiff files, latter case successfully converts 0/0
