# Author: Izie Wood (iw121@ic.ac.uk)
# Script: TreeHeight.R
# Desc: Loads tree data from CSV and calculates heights. 
# Writes results to csv file containing all original tree data and heights.
# Date: Oct 2021

## Imports ##
require(tidyverse)

## Functions ##
# load data
tree.data <- as.data.frame(read.csv("../data/trees.csv")) # Load as df
  
get_height <- function(degrees, distance){ 
  # Calculates the heights of trees given distance fo each tree
  # from its base and angle to its top, using the trigonometric formula
    radians <- degrees * pi / 180 
    height <- distance * tan(radians)
    return(height)  # returns height
}

# Get heights for tree data
# Create Height.m column as product of get_height using Angle and distance
tree.data <- tree.data %>% mutate(Height.m = get_height(Angle.degrees, Distance.m))

# Save data to CSV
location <- "../results/TreeHts.csv"
write.csv(tree.data, location)

# Update user
cat("Done! File saved to:", location)