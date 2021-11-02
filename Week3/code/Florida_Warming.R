#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Script: Florida.R
# Desc: Script calculating and presenting correlation coefficient between
# temperatures measured in the Florida Keys over the last century, to answer
# question: Is Florida getting warmer?
# 
# Date: Oct 2021

######### Open data set ##########

load("../data/KeyWestAnnualMeanTemperature.RData", verbose= TRUE) # ats


### Plot data ###
# save to pdf

pdf("../results/FloridaPlot.pdf", 11.7, 8.3)
plot(ats, xlab = "Years", ylab = "Temperature C", 
     main = "Key West temperatures over time",
     cex.lab = 1.5, cex.main = 1.5)
graphics.off()


### Calculate correlation coefficient of observed data 
Observed.Cor <- cor(ats$Year, ats$Temp) # Default Pearson's correlation coefficient

###### Conduct permutation analysis #######
# Compare correlation coefficient of observed data set against
# truly random distribution of correlation coefficients. 

num <- 5000 # number of permutations (reshuffling)
COR <- vector(,num) # Pre-allocate vector to hold correlation coefficients 

### Shuffle temperatures num times, and calculate correlation coefficient 
# for each random sample. 
# Save results to vector. 

#### function for sampling ####
data <- ats$Temp
size <- length(ats$Temp)

shuffle <- function(data, size){
  s <- sample(data, size, replace = TRUE) # sample data
  return(cor(ats$Year,s))
}

### Apply function to data set num times and save to vector ###
COR <- sapply(1:num, function(i) shuffle(data, size))

### plot results ###
# plot distribution of random correlation coefficients and save pdf
pdf("../results/permutationplot.pdf", 11.7, 8.3) # open pdf
    plot(COR, dnorm(COR, mean(COR), sd(COR)), 
     xlim=c(-0.55, 0.55), xlab = "Correlation coefficients of random samples",
     ylab = "Frequency", main="Distribution of random correlation coefficients", cex.lab=1.5, cex.main=1.5) # plot covariance of random samples

    abline(v=Observed.Cor, lwd=2, col = "red") # add line for observed covariance
    text(0.40, 3, "Observed", col = "red", cex=2) #label 
graphics.off() # close pdf


