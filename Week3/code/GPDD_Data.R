#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Script: Creates a world map using maps packages
# superimposes all the locations from which we have data 
# in the GPDD dataframe
# Date: Nov 2021

############## Load maps ###############
require(maps) # contains geographical outlines of many maps
require(tidyverse)

############# Load data ################
load("../data/GPDDFiltered.RData", verbose = TRUE) # gpdd


############# plot map ###################
# load exported world map from maps, save to df
world <- map_data("world") #ggplot2 function turns points in outline into df

# plot world map 
world.map <- ggplot() + geom_polygon(data= world, aes(x=long, y = lat,
                                         group = group), fill = "seashell3", 
                        colour = "lightslategrey") +
  coord_fixed(1.3) # cartesian coordinates- fixes aspect size ratio

# plot coordinates from gpdd
world.map <- world.map + geom_point(data = gpdd, aes(x = long, y = lat), 
                       colour = "yellow", size = 1)

print(world.map)

## data here seems heavily biased towards locations in the global
# north. Likely as a result of accessibility. This omits citings in large 
# parts of the world. 