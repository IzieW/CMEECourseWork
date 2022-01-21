#!/usr/bin/Rscript -vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Desc: Plots and saves three figures to files:
#         Distribution of Predator Mass by feeding interaction
#         Distribution of Prey Mass by feeding interaction
#         Distribution of Prey Mass/Predator Mass ratio by feeding interaction
#       and saves mean and median of each to a csv. 
# Arguments <- none 
# Date: Nov 2021

########## Load tidyverse ##########
require(tidyverse)

############ Load data #############
MyDF <- read.csv("../data/EcolArchives-E089-51-D1.csv")
MyDF <- mutate(MyDF, Type.of.feeding.interaction = 
                 as.factor(Type.of.feeding.interaction)) #set to factor

########### Vectors ###########
interaction.type <- unique(MyDF$Type.of.feeding.interaction) # save feeding types to vector

colour <- seq(1,20,1) # colour bank 

#### Predator mass distribution by feeding interaction ####
PredHist <- function(i){ 
  DFi <- filter(MyDF, Type.of.feeding.interaction == i)
  hist(log10(DFi$Predator.mass), xlab = "log10(Predator.mass (g))",
       main= i, col = sample(colour,1), breaks = 15)
}

pdf("../results/Pred_Subplots.pdf", 11.7, 8.3) # open file 

par(mfrow=c(3,2)) # create subplots 

sapply(interaction.type, FUN = PredHist)# apply plotting function to interaction types

graphics.off() # close file

#### Prey mass distribution by feeding interaction ####
PreyHist <- function(i){ 
  DFi <- filter(MyDF, Type.of.feeding.interaction == i)
  hist(log10(DFi$Prey.mass), xlab = "log10(Prey.Mass (g))",
       main= i, col = sample(colour,1), breaks = 15) # sample from colour bank quicker than rnorm
} 

pdf("../results/Prey_Subplots.pdf", 11.7, 8.3) # open file 

par(mfrow=c(3,2)) # create subplots 

sapply(interaction.type, FUN = PreyHist)# apply plotting function to interaction types

graphics.off() # close file

##### Prey/Predator Mass ratio distribution by feeding interaction #####

### calculate size ratios and save to vector
Size.ratio <- vector("numeric", length(MyDF$Prey.mass)) # pre-allocate

ratio <- function(i){ # function to calculate size ratio
  MyDF$Prey.mass[i]/MyDF$Predator.mass[i]
}

Size.ratio <- sapply(1:length(MyDF$Prey.mass), function(i) ratio(i)) #apply function for each row in MyDF

### add ratios to data frame 

DFs <- data.frame(MyDF, Size.ratio)

### plot 
RatioHist <- function(i){ 
  DFi <- filter(DFs, Type.of.feeding.interaction == i)
  hist(log10(DFi$Size.ratio), xlab = "log10(Size.ratio (g))",
       main= i, col = sample(colour,1), breaks = 15) # sample from colour bank quicker than rnorm
} 

pdf("../results/SizeRatio_Subplots.pdf", 11.7, 8.3) # open file 

par(mfrow=c(3,2))

sapply(interaction.type, FUN=RatioHist)

graphics.off() # close file

########### CSV mean, median ###########

DFV <- data.frame(DFs$Type.of.feeding.interaction, DFs$Predator.mass, 
                  DFs$Prey.mass, DFs$Size.ratio) # Isolate columns
colnames(DFV) <- noquote(c("Type.of.feeding.interaction",
                           "Predator.mass", "Prey.mass", "Size.ratio")) # save without quotation

# calculate mean, median for samples per interaction type
# and save to DF 
MMDF <- function(i, n){
  DFi <- filter(DFV, Type.of.feeding.interaction == i)
  Mean <- mean(DFi[[n]])
  Median <- median(DFi[[n]])
  Type.of.feeding.interaction <- i 
  Sample <- colnames(DFi[n])
  if (n == 2 & i == "predacious/piscivorous"){ # for first run, make new DF
    mainDF <<- data.frame(Sample, Type.of.feeding.interaction, Mean, Median)
  } else { #then, append to existing DF
    DFn <- data.frame(Sample, Type.of.feeding.interaction, Mean, Median)
    mainDF <<- rbind(mainDF, DFn)
  }
} 
# Apply function to interaction types, repeat for numbers 2, 3, 4
for (n in 2:4){
  sapply(interaction.type, FUN=MMDF, n)
}

# Save results to csv
write.csv(mainDF, "../results/PP_Results.csv")