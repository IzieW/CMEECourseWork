#!/bin/bash
for f in *.tif; # Note tif files must be in working directory
    do
        echo "Converting $f";
        convert "$f" "$(basename "$f" .tif).png";
    done 