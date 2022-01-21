#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Desc: Simple script illustrating break arguments in R
# Date: Oct 2021

i <- 0 # Initialize i 
    while(i < Inf){  # While i is less than infinity... so forever
        if (i == 10) {  # When i = 10, break from loop
            break
        }
        else {
            cat("i equals", i, "\n")
            i <- i + 1 # Update i
        }
    }
print("Loop finsihed!")
