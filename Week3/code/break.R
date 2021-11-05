#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Desc: Simple script illustrating break arguments in R
# Date: Oct 2021

i <- 0 # Initialize i 
    while(i < Inf){
        if (i == 10) { 
            break
        } # Break out of the while loop
        else {
            cat("i equals", i, "\n")
            i <- i + 1 # Update i
        }
    }