############################### Mini Project ##############################
#################### Work flow 2/4: Find Starting Values ##################

#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Date: Nov 2021
# Arguments <- none 
# Desc: Finds best starting values to use for NL models, calculates starting values and AIC

############## imports #############
require(minpack.lm)

############ Load Data #############

Data <- read.csv("../results/PreparedLogisticGrowthData.csv", stringsAsFactors = T) # import prepared data

IDs <- unique(Data$ID) #Create vector of unique IDs


############# Gompertz model ##############

# Copied from TheMulQuaBio: Fitting NonLinear LeastSquares
gompertz_model <- function(t, r_max, K, N_0, t_lag){ # "Modified gompertz growth model (Zwietering 1990)"
  return(N_0 + (K - N_0) * exp(-exp(r_max * exp(1) * (t_lag - t)/((K - N_0) * log(10)) + 1)))
} # Function predicts microbial growth
      # t = time
      # r_max = max growth rate
      # K = carrying capacity (ie. max abundance)
      # N_0 = Initial pop abundance 
      # t_lag = duration of "lag" phase

Calculate_rmax <- function(dataset){ #finds growth rate by fitting linear model
  model <- lm(log(PopBio) ~ Time, data=dataset)
  return(model$coefficients[2]) # Returns slope, ie. growth rate
}

Calculate_parameters <- function(dataset){
  r_max <- as.numeric(Calculate_rmax(dataset))
  K     <- max(log(dataset$PopBio)) # MulQuaBio
  N_0   <- min(log(dataset$PopBio)) # MulQuaBio
  t_lag <- dataset$Time[which.max(diff(diff(log(dataset$PopBio))))] # From MulQuaBio: Calculates max difference in differences to find point where exponential phase starts 
  
  return(c(r_max, K, N_0, t_lag))
}

Start_values <- data.frame(r_max = rep(NA, 277), K = rep(NA, 277), N_0 = rep(NA, 277), t_lag = rep(NA, 277))
for (i in 1:length(IDs)){
Start_values[i,] <- Calculate_parameters(filter(Data, ID == IDs[i]))
}

fit_gompertz <- function(dataset, start_values){
  r_max <- start_values[1]
  K     <- start_values[2]
  N_0   <- start_values[3]
  t_lag <- start_values[4]
  
  model <- nlsLM(log(PopBio) ~ gompertz_model(t = Time, r_max, K, N_0, t_lag),
                 data=dataset, start=list(r_max=r_max, K=K, N_0=N_0, t_lag=t_lag)
  )
}

Find_start_AIC <- function(i) { ### get ideal starting values for each population
  dataset <- Data %>% filter(ID==i) # isolate population
  start_values <- Calculate_parameters(dataset) # calculate start values
  gz_model <- fit_gompertz(dataset, start_values) #fit start values to nls() to find best convergence 
  
  return(as.numeric(AIC(gz_model))) #  return model AIC
}

AIC <- sapply(1:277, function(i) try(Find_start_AIC(i), silent = TRUE)) # get best start values for each dataset
AIC <- as.numeric(AIC)

#ValuesDF <- data.frame(r_max = rep(NA, length(IDs)), K = rep(NA, length(IDs)), N_0 = rep(NA, length(IDs)), 
                                                                              #  t_lag = rep(NA, length(IDs)))
#for (i in 1:277){
 # ValuesDF[i,] <- StartValues[[i]] #transport StartValues to DF
#}

AICDF <- bind_cols(ID = IDs, Model = "Gompertz", AIC = AIC) #combine start values with ID

AICDF <- filter(AICDF, !is.na(as.numeric(AIC))) #filter out any non-numeric error warnings

########## Save starting values to CSV ##############
write.csv(AICDF, "../results/GompertzBestFit.csv")
write.csv(Start_values, "../results/StartValuesGompertz.csv")

