###################### Mini Project #######################
################ Work flow 1/4: Data Prep #################

#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Date: Nov 2021
# Arguments <- none 
# Desc: Prepares data for further analysis, saves to csv:
  # Creates unique IDs for identifying data sets
  # Deals with missing data and problematic values
  # Saves modified data to CSV

################ Imports ################
require(tidyverse) # for data organisation
require(crayon) # for colouring outputs

############### Load data ###############
Data <- read.csv("../data/LogisticGrowthData.csv") # load data, save to variable
MetaData <- read.csv("../data/LogisticGrowthMetaData.csv") # load meta data
cat("Loading data from ../data/LogisticGrowthData.csv")

############## Prepare data #############
### Create IDs to distinguish unique populations
Data$ID <- paste(Data$Species, Data$Temp, Data$Medium, 
                 Data$Citation, sep = "_") # Concatenate Species, Temp, Medium + Citation

Data$ID <- Data %>% group_by(ID) %>% group_indices() # group by ID, return integer for each group

cat("\nGrouping ")
cat(length(unique(Data$ID)))
cat(" unique populations...")

### Remove problematic data points
# Negative values for time or population/biomass
Data <- Data %>% filter(Time >= 0 & PopBio > 0) #population size/Biomass must be greater than zero
cat("\nRemoving problematic values..")



############ Save Data to csv ############
write.csv(Data, "../results/PreparedLogisticGrowthData.csv")
cat("\nFinished! Prepared data saved to ../results")


