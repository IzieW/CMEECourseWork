#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Desc: Simple script illustrating control functions in R 
# Date: Oct 2021

a <- TRUE
if (a == TRUE){
    print ("a is TRUE")
} else {
    print ("a is FALSE")
}

z <- runif(1) ## Generate uniformly distributed random number
if (z <= 0.05) {
    print ("Less than a half")
    }

for (i in 1:10){
    j <- i * i
    print(paste(i, " squared is", j ))
}

for(species in c('Helidoxa rubinoides',
                'Boissonneaua jardini',
                'Sula nebouxii')){
    print(paste('The species is', species))
}

v1 <- c("a","bc","def")
for (i in v1){
    print(i)
}

i <- 0
while (i <10){
    i <- i + 1
    print(i^2)
}