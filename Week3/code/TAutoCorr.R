#!/usr/bin/Rscript --vanilla
# Author: DashingDingos
# Desc: Are temperatures of one year significantly correlated with the next year? 
# Computes correlation between successive years using permutation test
# Date: Jan 2022

############ imports #############
require(tidyverse)

## Load data
load("../data/KeyWestAnnualMeanTemperature.RData", verbose= TRUE)  # ats

########## Functions ############
get_lineup <- function(ats) {
  # Align n-1 pairs of temperatures
  temp1 <- ats %>% filter(as.integer(Year) != 2000) %>% select(Temp)
  temp2 <- ats %>% filter(as.integer(Year) != 1901) %>% select(Temp)
  df <- data.frame(temp1, temp2)
  return(df)
}

## Save correlation coefficient between n-1 pairs of years
df <- get_lineup(ats) # Line up n-1 pairs 
observed_cor <- cor(df$Temp, df$Temp.1) # Calculate correlation coefficient

## Create a plot of the data with the correlation shown
## This figure is used in the report, so this code should be run first!
pdf("../results/graph.pdf")
plot(
  df$Temp, df$Temp.1,
  main="Correlation of temperature per year with next year",
  xlab="Temperature per year (°C)",
  ylab="Temperature of next sucessive years (n-1) (°C)",
  col="black",
  pch=3
)
abline(lm(Temp~Temp.1, data=df),col='blue')
legend(24.0,26.0, legend="Year",col="black", pch=3)
graphics.off()

### Permutation analysis ###
shuffle_data <- function(df, size, withplot=FALSE) {
  # Shuffle data, return correlation coefficient
  shuffled_temps <- sample(df$Temp.1, size, replace = FALSE)
  return(cor(df$Temp, shuffled_temps))
}

set.seed(1234) # set seed for reproducibility
random_cor <- vector(,10000) # Pre-allocate vector
size <- length(df$Temp)
random_cor <- sapply(1:10000, function(i) shuffle_data(df, size))

## Plot results & save to file
pdf("../results/graph2.pdf", 11.7, 8.3)
hist(
  random_cor,
  xlab="Correlation Coefficent",
  ylab="Frequency",
  main="Histagram of coefficent between n-1 pairs of years",
  xlim=c(-0.4,0.4),ylim=c(0,2100)
)
abline(
  v=observed_cor, lwd=2, col = "red", lty=2
) # add line for observed covariance
text(
  0.24, 1700, "Observed correlation\n coefficient", col = "red", cex=1
) #label 
graphics.off()

## Calculate what percentage of random correlation coefficients 
# are larger than the observed correlation coefficient 
pvalue <- (
  as.data.frame(random_cor)
  %>% filter(random_cor > observed_cor)
  %>% count()
) / length(random_cor) 
print(paste(
  "The correlation coefficient is: ",
  observed_cor, " (p = ", pvalue, ")", sep = ""
))
