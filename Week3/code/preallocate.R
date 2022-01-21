#!/usr/bin/Rscript --vanilla
# Author: Izie Wood
# Desc: Script demonstrating use of memory when pre-allocating vectors vs. not 
# Date: Oct 2021

# Without pre-allocation
NoPreallocFun <- function(x){
    a <- vector() # empty vector
    for (i in 1:x) {
        a <- c(a, i)  # vector is updated + given new space in memory each time
        print(a)
        print(object.size(a))  # Print memory requirements (changes with every loop)
    }
}

system.time(NoPreallocFun(10))

# With pre-allocation
PreallocFun <- function(x){
    a <- rep(NA, x) # pre-allocated vector
    for (i in 1:x) {
        a[i] <- i  # populates space in existent memory
        print(a)
        print(object.size(a)) # Memory requirements stay the same
    }
}

system.time(PreallocFun(10))
