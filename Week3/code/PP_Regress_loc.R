#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Desc: Plots and saves regression lines to pdf
# Saves regression results to csv
# Date: Nov 2021

############### Load packages ##############
require(ggplot2)
require(tidyverse)

################ Load data #################
MyDF <- read.csv("../data/EcolArchives-E089-51-D1.csv")

# set factors
MyDF <- mutate(MyDF, Type.of.feeding.interaction = 
                 as.factor(Type.of.feeding.interaction),
               Predator.lifestage = as.factor(Predator.lifestage))

################## Calculate regression results ################
## group data 
t <- MyDF %>% nest(data = -Type.of.feeding.interaction) %>% # create nest of all data minus feeding interaction, grouped by type of feeding interaction
mutate(data = map(data, ~.x %>% nest(data = -Predator.lifestage))) # create nest within feeding interaction, grouped by predator lifestage

d <- t %>% unnest() # remove first nest- output is data grouped by both factors

# perform linear model for each group in d 
regress <- d %>% mutate( fit = map(data, ~lm(Prey.mass ~ Predator.mass, data = .x)),#apply lm() to each value in grouped data
                         tidied = map(fit, broom::tidy),
                         glance = map(fit, broom::glance)) #tidy and glance to clean up output of lm

### Gather results for table 
coef <- regress %>% unnest(tidied) %>% select(term, estimate) %>% #select coefficients
  pivot_wider(names_from = term, values_from = estimate) %>% #reshape data
  unnest() 

colnames(coef) <- c("intercept", "slope") #rename columns

g <- regress %>% unnest(glance) %>% select(r.squared, F.value = statistic, p.value) # select statistical tests

df_regress <- bind_cols(regress[,1:2], coef, g) #bind columns into table

### save results to csv
write.csv(df_regress, "../results/PP_Regress_Results.csv")




