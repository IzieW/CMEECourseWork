#!/usr/bin/Rscript --vanilla
# Author: Izie Wood
# Desc: Script demonstrating different types of vectorisation using a sampling experiment
# Date: Oct 2021

## A function to take a sample size of n from a population "popn" and return its mean 
myexperiment <- function(popn,n){
    pop_sample <- sample(popn, n, replace = FALSE) # replace = FALSE- numbers not put back into sample size after being picked
    return(mean(pop_sample))
}

## Calculate means using a for loop without pre-allocation: 
loopy_sample1 <- function(popn, n, num){
    result1 <- vector() #Initialise empty vector of size 1
    for (i in 1:num){
        result1 <- c(result1, myexperiment(popn, n))
    }
    return(result1)
}

## To run "num" iterations of the experiment using a for loop on a vector with pre-allocation:
loopy_sample2 <- function(popn, n, num){
    result2 <- vector(,num) #pre-allocate expected size 
    for(i in 1:num){
        result2[i] <- myexperiment(popn, n)
    }
    return(result2)
}

## To run "num" iterations of the experiment using a for loop on a list with pre-allocation
loopy_sample3 <- function(popn, n, num){
    result3 <- vector("list", num) # pre-allocate expected size
    for(i in 1:num){
        result3[[i]] <- myexperiment(popn, n)
    }
    return(result3)
}

## To run "num" iterations of the experiment using vectorisation with lapply:
lapply_sample <- function(popn, n, num){
    result4 <- lapply(1:num, function(i) myexperiment(popn, n))
    return(result4)
}

## To run "num" iterations of the experiment using vectorization with sapply: 
    sapply_sample <- function(popn, n, num){
        result5 <- lapply(1:num, function(i) myexperiment(popn,n))
        return(result5)
    }

popn <- rnorm(1000) # generate a population
hist(popn)

## Run and time different functions

n <- 20 # sampe size for each experiment
num <- 1000 # Number of times to rerun the experiment

print("The loopy, none-pre-allocation approach takes:" )
print(system.time(loopy_sample1(popn, n, num)))

print("The loopy, but with pre-allocation approach takes:" )
print(system.time(loopy_sample2(popn, n, num)))

print("The loopy, non-pre-allocation approach on a list takes:" )
print(system.time(loopy_sample3(popn, n, num)))

print("THe vectorized sapply approach takes:" )
print(system.time(sapply_sample(popn, n, num)))

print("The vectorized lapply approach takes:" )
print(system.time(lapply_sample(popn, n, num)))
