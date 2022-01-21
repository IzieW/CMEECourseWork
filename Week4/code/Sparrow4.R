#!/usr/bin/Rscript -- vanilla 

########### Exercises ##############

## 1. Calculate SE of Tarsus, Mass, Wing and Bill length
# note N of each. 
# Subset data to only 2001, calculate SE for the above in
# 2001. 
# Calculate the 95% CI for each mean. 

### Organize data into vector ####
d <- read.table("../data/SparrowSize.txt", header = TRUE) # load data

dvect <- vector("list", 4) # preallocate vector

for (i in 1:4){ # populate vector with Tarsus, Bill, Wing, Mass 
  dvect[[i]] <- d[,i+2][!is.na(d[,i+2])]
  }

###################### SE ##############################
SEvect <- vector("list", 4)

SE <- function(x){ # function for Standard Error
  sqrt(var(x)/length(x))
}

SEvect <- sapply(dvect, FUN = SE) # Apply SE function to items in vector

names <- c(names(d))
print("The standard error for measurements in SparrowSize.txt are")
for (i in 1:4){
  print(paste(names[i+2],": ",SEvect[[i]]))
}

#################### SE in 2001 #########################
d2001 <- subset(d, d$Year==2001) # segment data from 2001
  
dvect2001 <- vector("list", 4) # preallocate vector

for (i in 1:4){ # populate vector with Tarsus, Bill, Wing, Mass 
  dvect2001[[i]] <- d2001[,i+2][!is.na(d2001[,i+2])] # populate measurements from 2001
}

SEvect2001 <- sapply(dvect2001, FUN = SE) # apply SE function

print("The standard error for measurements in 2001 are")
for (i in 1:4){
  print(paste(names[i+2],": ",SEvect2001[[i]]))
}

################# Confidence interval ##################
dCI <- vector("list",4)
 for (x in 1:4){
  m <- mean(dvect[[x]])
  SE <- sqrt(var(dvect[[x]])/length(dvect[[x]]))
  up <- m + SE
  down <- m - SE
  dCI[[x]] <- c(down, up)
 }
print("The confidence interval for measurements in 2001 are")
for (i in 1:4){
  print(paste(names[i+2],":"))
  print(dCI[[i]])
}




