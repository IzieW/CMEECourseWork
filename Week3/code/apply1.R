#!/usr/bin/Rscript --vanilla
# Author: Izie Wood
# Desc: Script demonstrating use of apply
# Date: Oct 2021

## Build a random matrix 
M <- matrix(rnorm(100), 10, 10)  # 10 x 10 matrix

## Take the mean of each row
RowMeans <- apply(M, 1, mean)  # apply across rows
print (RowMeans)

## Now the variance
RowVars <- apply(M, 1, var)
print (RowVars)

## By column
ColMeans <- apply(M, 2, mean)  # apply across columns
print (ColMeans)
