#!/usr/bin/env Rscript

# A script that finds the heights of provided tree data. Outputs to target
# file, or the results directory with a helpful name if not provided.
# Usage: ./get_TreeHeight.R input [output]


# This function calculates heights of trees given distance of each tree
# from its base and angle to its top, using the trigonometric formula.
#
# height = distance * tan(radians)
#
# ARGUMENTS
# degrees:  the angle of elevation to the top of the tree
# distance: the distance from base of tree (e.g., meters)
#
# OUTPUT
# The heights of the tree, same units as "distance"
TreeHeight <- function(degrees, distance){
  radians <- degrees * pi/180 #Equation 1
  height <- distance * tan(radians) #Main Equation
  print(paste("Tree height is:", height))
  
  return(height) 
}

# Fetch the parameters (excluding the binary name)
#
# The function commandArgs() extracts all the command line arguments and returns them as a vector.
# Since the first elements of the vector are always the same, we can tell commandArgs() to only return
# the arguments that come after the calling of the R script.
argv <- commandArgs(trailingOnly=TRUE)
inp <- argv[1]
out <- argv[2]

# If the input file is not provided, use a default
if (is.na(inp)){
  inp <- "../data/trees.csv"
}

# Reading in the data, extracting the csv filename from index 1 of the command line arguments vector.
TreeData <- read.csv(inp, header = TRUE)

# Tell the user we're starting
print("Calculating tree heights...")

# Process the data with the function above.
TreeData$Height.m <- TreeHeight(TreeData$Angle.degrees, TreeData$Distance.m)

if (is.na(out)) {
  # Strip the path and (optional) file extension
  base <- basename(inp)
  # Remove extension to only file1=args[1]
  no_ext <- gsub(pattern = "\\..*$", "", base) 
  out <- paste('../results/', no_ext, '_treeheights.csv', sep='')
  warning(paste(
    'No output provided, outputting to', out
  ))
}

# Write to the output file
write.csv(TreeData, out, row.names = FALSE)

# Update user
print(paste("Done! File saved to:", out))

