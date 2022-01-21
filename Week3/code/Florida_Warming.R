#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Script: Florida.R
# Desc: Script calculating and plots correlation coefficient between
# temperatures measured in the Florida Keys over the last century, to answer
# question: Is Florida getting warmer?
# 
# Date: Oct 2021

######### Functions ##########

# Load in data
load("../data/KeyWestAnnualMeanTemperature.RData", verbose= TRUE) # ats

#### Calculate correlation coefficient ###
# Save correlation coefficient of observed data
observed_cor <- cor(ats$Year, ats$Temp)  # Correlation coefficient of observed data

###### Conduct permutation analysis #######
# Compare correlation coefficient of observed data set against
# truly random distribution of correlation coefficients. 

shuffle_once <- function(ats, size){
  # Shuffle temperatures randomly, return correlation coefficient
  s <- sample(ats$Temp, size, replace = FALSE) # sample each data point only once
  return(cor(ats$Year,s))
}

do_permutation <- function(ats) {
  # Return vector of correlation coefficients from data shuffled 5000 times
  num <- 5000  # assign number of permutations
  random_cor <- vector(,num)  # pre-allocate vector
  size <- length(ats$Temp)  # Get vector length once
  
  random_cor <- sapply(1:num, function(i) shuffle_once(ats, size))  # shuffle data num times, save results to vector
  
  return(random_cor)
} 

plot_permutation <- function(ats, observed_cor) {
  # Run permutation analysis and plot distribution of random correlation coefficients
  # save to pdf
  random_cor <- do_permutation(ats)  # Get random correlations from permutation analysis
  
  # plot results
  pdf("../results/Permutation_results.pdf", 11.7, 8.3) # open pdf
    hist(random_cor, 
     xlim=c(-0.65, 0.65), xlab = "Correlation coefficients of random samples",
     ylab = "Frequency", main="Distribution of random correlation coefficients", cex.lab=1.5, cex.main=1.5) # plot covariance of random samples

    abline(v=observed_cor, lwd=2, lty = 3, col = "red") # add line for observed covariance
    text(0.35, 800, "Observed correlation\n coefficient", col = "red", cex=1.5) #label 
   graphics.off() # close pdf
    
    cat("Permutation analysis finished,\nresults saved to ../results/Permutation_results.pdf")
}

plot_permutation(ats, observed_cor)


