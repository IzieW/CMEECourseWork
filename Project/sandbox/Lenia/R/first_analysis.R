# !/usr/bin/Rscript 

# First analysis of exciting mutation
# seed 0, fixation 10

require("tidyverse")

times = read.csv("../results/fixation_10_seed_0_times.csv", header = T)

generations <- seq(1, length(times[,1]), 1)

times["time"] <- generations

plot(times$time, times$wild, type="l")

generations <- seq(1, length(times2[,1]), 1)
times2 <- read.csv("../results/fixation_25_seed_0_times.csv", header = T)
plot(generations, times2$wild, type="l", col="blue")