#!/usr/bin/Rscript --vanilla
# Author: Izie Wood (iw121@ic.ac.uk)
# Script: Vectorize2.R
# Desc: Re-write stochastic Ricker equation to make better vectorized
# Date: Oct 2021 

# Runs the stochastic Ricker equation with gaussian fluctuations

rm(list = ls())  # clear environment

stochrick <- function(p0 = runif(1000, .5, 1.5), r = 1.2, K = 1, sigma = 0.2, numyears = 100)
# runif 1000 random numbers from uniform .5, 1.5
{

  N <- matrix(NA, numyears, length(p0))  #initialize empty matrix

  N[1, ] <- p0 #populate row 1 with random numbers from P0, aka population 0 

  for (pop in 1:length(p0)) { #loop through the populations

    for (yr in 2:numyears){ #for each pop, loop through the years

      N[yr, pop] <- N[yr-1, pop] * exp(r * (1 - N[yr - 1, pop] / K) + rnorm(1, 0, sigma)) # add one fluctuation from normal distribution
    
     }
  
  }
 return(N)
}
print("Pre-allocated model takes:")
print(system.time(res2<-stochrick()))

# Now write another function called stochrickvect that vectorizes the above to
# the extent possible, with improved performance: 

rm(list = ls())

stochrickvect <- function(p0 = runif(1000, .5, 1.5), r = 1.2, K = 1, sigma = 0.2,numyears = 100)
{

  N <- matrix(NA, numyears, length(p0))  #initialize empty matrix

  N[1, ] <- p0 #populate row 1 with random numbers from P0, aka population 0 

  for (yr in 2:numyears) { #loop through the populations
      N[yr,] <- N[yr-1,] * exp(r * (1 - N[yr - 1,] / K) + rnorm(1000, 0, sigma)) # add one fluctuation from normal distribution
     }

 return(N)
}

print("Vectorized Stochastic Ricker takes:")
print(system.time(res2<-stochrickvect()))