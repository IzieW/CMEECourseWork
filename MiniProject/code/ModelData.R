########################### Mini Project #############################
##################### Workflow 3/4: Fits models ######################

#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Date: Nov 2021
# Arguments <- none 
# Desc: Fits linear models and calculates AIC

############### Load Data ###############
Data <- read.csv("../results/PreparedLogisticGrowthData.csv", stringsAsFactors = TRUE)

Data$LogPopBio <- log(Data$PopBio) #Log Transform PopBio

IDs <- unique(Data$ID) # Save population IDs to vector

############## Fit linear ################
# Fits quadratic model for x data set, returns AIC
Fit_Quadratic <- function(x){
  Popq <- Data %>% filter(ID == x) #Isolate population with select ID from Data
  QuaModel <- lm(LogPopBio ~ poly(Time, 2), data = Popq) #fit quadratic model to population
  return(AIC(QuaModel)) # Calculate IC
}
# Fits Cubic model for x data set, returns AIC
Fit_Cubic <- function(x){
  Popc <- Data %>% filter(ID == x) # Isolate population
  CubModel <- lm(LogPopBio ~ poly(Time, 3), data = Popc)
  return(AIC(CubModel))
}

# fit models to each dataset
Quadratic_AIC <- sapply(IDs, FUN = Fit_Quadratic) # fit quadratic model to each population
Cubic_AIC <- sapply(IDs, FUN = Fit_Cubic) # fit Cubic model to each Population

# Add IDs 
Quadratic_best_fit <- bind_cols(ID = IDs, Model = "Quadratic", AIC = Quadratic_AIC)
Cubic_best_fit <- bind_cols(ID = IDs, Model = "Cubic", AIC = Cubic_AIC)


#################### return results ##############
# Save results to CSV
write.csv(Cubic_best_fit, "../results/Cubic.csv")
write.csv(Quadratic_best_fit, "../results/Quadratic.csv")