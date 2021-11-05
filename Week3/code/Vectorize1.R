#!/usr/bin/Rscript --vanilla
# Author: Izie Wood
# Desc: Script demonstrating time taken for vectorized vs. non-vectorized functions
# Date: Oct 2021

M <- matrix(runif(1000000),1000,1000)

sumALLElements <- function(M){
    Dimensions <- dim(M)
    Tot <- 0
    for (i in 1:Dimensions[1]){
        for (j in 1:Dimensions[2]){
            Tot <- Tot + M[i,j]
        }
    }
    return (Tot)
}
print("Using loops, the time taken is:")
print(system.time(sumALLElements(M)))

print("Using the in-built vectorized function, the time taken is:")
print(system.time(sum(M)))