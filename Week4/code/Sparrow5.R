#!/usr/bin/Rscript --vanilla 
# Stats week, Monday (5) exercises

d <- read.table("../data/SparrowSize.txt", header = TRUE) #Load data


###### 1. Test if wing length in 2001 differs from 
# the grand total mean. 

d2001 <- subset(d, d$Year == 2001) # isolate 2001

dall <- sample(d$Wing, length(d2001$Wing), replace = FALSE) # sample grand total


t.testwing <- t.test(dall,d2001$Wing)

t.testwing

### 1. Test is male and female wing length differ from 2001

t.testsex <- t.test(d2001$Wing~d2001$Sex.1)

t.testsex

### 2. Run a batch test: check for each year whether any measures
# differ from the grand total mean

m <- vector("list", 4) # preallocate mean vector

years <- unique(d$Year)
MeanYear <- data.frame(NA, le
MeanYear[,1] <- 2000:2010
names(MeanYear) <- c("Year", "Tarsus", "Bill", "Wing", "Mass")

for (i in 1:4){
  for (yr in range(d$Year)){
    m[[i]] <- mean(d[,i+2])
    
  }
  
}











