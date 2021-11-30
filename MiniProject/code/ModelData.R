########################### Mini Project #############################
##################### Workflow 2/4: Fits models ######################

#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Date: Nov 2021
# Arguments <- none 
# Desc: Fits models and calculates AIC

################ Imports ################
require(tidyverse)

############### Load Data ###############
Data <- read.csv("../results/PreparedLogisticGrowthData.csv", stringsAsFactors = TRUE)

############## Functions #################
IDs <- unique(Data$ID) # Save population IDs to vector

# Fits quadratic model for x data set, returns AIC
FitQuaModel <- function(x){
  Popq <- Data %>% filter(ID == x) #Isolate population with select ID from Data
  QuaModel <- lm(PopBio ~ poly(Time, 2), data = Popq) #fit quadratic model to population
  return(AIC(QuaModel)) # Calculate IC
}
# Fits Cubic model for x data set, returns AIC
FitCubModel <- function(x){
  Popc <- Data %>% filter(ID == x) # Isolate population
  CubModel <- lm(PopBio ~ poly(Time, 3), data = Popc)
  return(AIC(CubModel))
}

# Fit quadratic model to each population in Data 
QuaModel_AIC <- sapply(IDs, FUN = FitQuaModel) # fit quadratic model to each population
CubModel_AIC <- sapply(IDs, FUN = FitCubModel) # fit Cubic model to each Population

# Save results to tibble
Models_AIC <- bind_cols(ID = IDs, QuaModel = QuaModel_AIC, CubModel = CubModel_AIC) # combines vectors

# Save results to CSV
write.csv(Models_AIC, "../results/Models_AIC.csv")