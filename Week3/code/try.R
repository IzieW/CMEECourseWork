#!/usr/bin/Rscript --vanilla
# Author: Izie Wood
# Desc: Script illustrates examples of try
# Date: Oct 2021

doit <- function(x){ #runs simulation by sampling from synthetic population
    temp_x <- sample(x, replace = TRUE)
    if(length(unique(temp_x)) > 30) { #only take mean if sample was sufficient
    print(paste("Mean of this simple was:", as.character(mean(temp_x)))) # returns mean
    }
    else {
        stop("Couldn't calculate mean: too few unique values!")
    }
}

set.seed(1345) # again, to get the same result for illustration

popn <- rnorm(50)

hist(popn)

## Without try
lapply(1:15, function(i) doit(popn))

## With try
result <- lapply(1:15, function(i) try(doit(popn), FALSE))
class(result)
result

# The same using loop
result <- vector("list", 15) #Preallocate/Initialize
for(i in 1:15) {
    result[[i]] <- try(doit(popn), FALSE)
}
