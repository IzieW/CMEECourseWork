# This function calculates the heights of trees given distance fo each tree
# from its base and angle to its top, using the trionometric formula
# 
# height = distance * tan(radians)
#
# ARGUMENTS
# degrees: The angle of elevation of the tree
# distance: The distance from the base of the tree (e.g., meters)
#

# Read csv file, save as contents as data frame
CSTree <- read.csv("../data/trees.csv") # pun on csv 

# OUTPUT calculate tree height from distance and degree
# The heights of the tree, same units as "distance"
TreeHeight <- function(degrees, distance){
    radians <- degrees * pi / 180
    height <- distance * tan(radians)

for (i in CSTree){
    
}
    print(paste("Tree height is:", height))

    return(height)
}

TreeHeight(37, 40)