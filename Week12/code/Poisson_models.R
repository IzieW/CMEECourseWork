#!usr/bin/Rscript --vanilla
# GLM WEEK: Poisson Models

## Imports ##
require(tidyverse)
require(MASS)
require(ggpubr)

fish <- read.csv("../data/fisheries.csv", stringsAsFactors = T)

## Does total abundance change with mean depth? ##
p <- ggplot(fish, aes(x = MeanDepth, y = TotAbund)) + geom_point() + 
  labs(x = "Mean Depth (km)", y = "Total Abundance") + theme_classic()

print(p)

# Model equation: in(TotAbund) = B0 + B1 * MeanDepth

M1 <- glm(TotAbund~MeanDepth, data = fish, family = "poisson")
print(summary(M1))