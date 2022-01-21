#!/usr/bin/Rscript --vanilla 
# GLM Week: Linear Models Catch up 

require(tidyverse)
## Load Data
d <- read.table("../../Week4/data/SparrowSize.txt", header = TRUE)
d <- na.omit(d)

## Centrality and Spread
f <- qplot(d$Tarsus, bins = 10) + labs(x = "Sparrow Tarsus length (mm)")

# Use density plot to see probabilities
p <- qplot(d$Tarsus, geom = "density") + labs(x = "Sparrow Tarsus length (mm)")
p <- p + geom_vline(xintercept = mean(d$Tarsus), colour = I("red"))
print(p) # Notice two peaks around mean... 

# Could be difference in sex...
print(t.test(d$Tarsus~d$Sex)) # Shows two means, with significant difference between them

# Plot agaisnt sex... 
p <- qplot(Tarsus, data = d, geom = "density", colour = Sex.1) + labs(x = "Sparrow Tarsus length (mm") +
  geom_vline(xintercept = mean(d$Tarsus[d$Sex == 0])) + 
  geom_vline(xintercept = mean(d$Tarsus[d$Sex == 1]))

print(p) # See two means 

