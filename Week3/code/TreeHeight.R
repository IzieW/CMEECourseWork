# Author: Izie Wood (iw121@ic.ac.uk)
# Script: TreeHeight.R
# Desc: Calculates Tree height from distance and degree. 
# Loads data from CSV
# Creates csv output file containing original data and Tree height
#
# Date: Oct 2021


######## Load CSV file to data frame #########
CSTree <- (read.csv("../data/trees.csv")) # pun of csv

####### Calculate tree height ##########
# # This function calculates the heights of trees given distance fo each tree
# from its base and angle to its top, using the trionometric formula
# 
# height = distance * tan(radians)
#
# ARGUMENTS
# degrees: The angle of elevation of the tree
# distance: The distance from the base of the tree (e.g., meters)

TreeHeight <- function(degrees, distance){
    radians <- degrees * pi / 180
    height <- distance * tan(radians)
    return(height)
}

# Use above function for each row in data frame, save to vector
l <- length(CSTree[,1]) # column length/row numbers
Col4 <- vector(,l) # preallocate vector size
for (i in 1:l){
    TreeHts <- TreeHeight(CSTree[i,3], CSTree[i,2]) # Input degree/distance from dataframe
    Col4[i] <- TreeHts
}
####### Save heights as CSV ################
# Add heights to dataframe CSTree
CSTree[4] <- Col4
colnames(CSTree)[4] <- "Height.m" # name column

# Save dataframe to CSV
Location <- "../results/TreeHts.csv"
write.csv(CSTree, Location)

# Update user
print("Calculating tree heights...")
print(paste("Done! File saved to:", Location))