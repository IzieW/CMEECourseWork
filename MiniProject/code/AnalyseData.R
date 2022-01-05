###################### Mini Project #######################
################ Work flow 4/4cub: Analyse data ##############

#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Date: Nov 2021
# Arguments <- none 
# Desc: Import results and plot curves for each model
# Summarizes best fit for each

############# Imports ###############
require(tidyverse)
require(minpack.lm)

############ Load Data ##############
Data <- read.csv("../results/PreparedLogisticGrowthData.csv", stringsAsFactors = T)
Data$LogPopBio <- log(Data$PopBio)


Quadratic_fit <- read.csv("../results/Quadratic.csv")
Cubic_fit <- read.csv("../results/Cubic.csv")
Gompertz_fit <- read.csv("../results/GompertzBestFit.csv")

IDs <- unique(Data$ID)

startvals <- read.csv("../results/StartValuesGompertz.csv")

########### Gompertz ##############
Gompertz_model <- function(t, r_max, K, N_0, t_lag){ # modified model
  return(N_0 + (K - N_0) * exp(-exp(r_max * exp(1) * (t_lag - t)/((K - N_0) * log(10)) + 1)))
}

######## Analyse best fit ############
Model_Compare <- bind_rows(Gompertz_fit, Quadratic_fit, Cubic_fit) #combine three model fits into one

Gompertz_fit <- replace_na(Gompertz_fit, list(AIC = 100000000000)) #replace NAs with numeric quantity

Best_fits <- Model_Compare %>% group_by(ID) %>% filter(AIC == min(AIC))

Best_fits <- Best_fits %>% group_by(Model) %>% summarise(n = n()) # Find number of times each model won

################## Plot Models #################
Plot_all <- function(i){ #Plot each model to each dataset
  
 dataset <- filter(Data, ID == IDs[i])
 ID <- IDs[i]
  timepoints <- seq(min(dataset$Time), max(dataset$Time), 1)
  
  PredGompertz <- Gompertz_model(t = timepoints, r_max = startvals[i,]["r_max"],
                                 K = startvals[i,]["K"], N_0 = startvals[i,]["N_0"],
                                 t_lag = startvals[i,]["t_lag"])
  
  dfG <- data.frame(Model = "Gompertz", Time = timepoints, Pred = PredGompertz)
  
  QuaFit <- lm(LogPopBio ~ poly(Time, 2), data = dataset)
  PredQuadratic <- predict.lm(QuaFit, data.frame(Time = timepoints))
  
  dfQ <- data.frame(Model = "Quadratic", Time = timepoints, Pred = PredQuadratic)
  
  CubFit <- lm(LogPopBio ~ poly(Time, 3), data = dataset)
  PredCubic <- predict.lm(CubFit, data.frame(Time = timepoints))
  
  dfC <- data.frame(Model = "Cubic", Time = timepoints, Pred = PredCubic)
  
  df <- bind_rows(dfG, dfQ, dfC)
  
  p <- ggplot(dataset, aes(x = Time, y = LogPopBio)) + theme_bw() +
  theme(aspect.ratio = 1) + geom_point(size = 3) +
    labs(x = "Time (hours)", y = "log(PopBio)", title = ID) +
    geom_line(data = df, aes(x = Time, y = Pred, colour = Model))
  
  print(p)
}

sapply(1:277, function(i) try(Plot_all(i), silent = TRUE))

