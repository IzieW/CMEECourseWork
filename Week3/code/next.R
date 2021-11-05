#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Desc: Script illustrating next function in R
# Date: Oct 2021

for (i in 1:10) {
    if ((i %% 2 ) == 0) #check if number is odd
        next #pass to next iteration of loop
    print(i)
}