#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Desc:  Illustrates control flows in R. 
# Date: Oct 2021

# IF statements: 
a <- TRUE
if (a == TRUE) {
    print ("a is TRUE")
} else {
    print ("a is FALSE")
}

z <- runif(1) ## Generate uniformly distributed random number
if (z <= 0.05) {
    print ("Less than a half")
    }

# FOR loops:
for (i in 1:10) {  # for i in range 1-10
    j <- i * i
    print(paste(i, " squared is", j ))
}

# Just like python, can also loop over vector strings
for(species in c('Helidoxa rubinoides',
                'Boissonneaua jardini',
                'Sula nebouxii')){
    print(paste('The species is', species))
}

v1 <- c("a","bc","def")
for (i in v1) {
    print(i)
}

# WHILE loops: 
i <- 0
while (i <10 ) { #While is is less than 10, print i
    i <- i + 1
    print(i^2)
}
