# !/usr/bin/Rscript --vanilla

# ANALYSE RESULTS OF ORBIUM EVOLUTION IN ENVIRONMENTS OF DIFFERENT GRADIENTS
# Orbium evolved for 1.5 hours in environment of n=1, r=60 and gradients 
# from 0.5:9.5 of intensity, where gradient value = lambda in exponential
# distribution
#
#
# Script analyses time logs and the nature of mutations across different gradients

## IMPORTS ##
require(tidyverse)

##### LOAD DATA #######
gradients <- seq(0.5, 9.5, 1)

load_times <- function(){
  
  times <- read.csv("times/TIMELOG_orbiumt4200n1r60_gradient0.5.csv", header=TRUE)
  times <- as.data.frame(times[2:5])
  times["gradient"] <- 0.5
  times["runs"] <- seq(1, length(times[,1]))
  
  for (i in gradients[2:10]){
    filename <- paste("times/TIMELOG_orbiumt4200n1r60_gradient", i, ".csv", sep="")
    temp <- read.csv(filename, header=TRUE)
    temp <- as.data.frame(temp[2:5])
    temp["gradient"] <- i
    temp["runs"] <- seq(1, length(temp[,1]))
    times <- bind_rows(times, temp)
  }
  
  return(times)
}

get_end_times <- function(g){
  return (filter(times, gradient== g) %>% select(wild_mean) %>% max())
}

get_end_var <- function(g){
  return (as.numeric(filter(times, gradient== g) %>% filter(runs == max(runs)) %>% select(wild_var) %>% sqrt()))
}


########## ANALYSE TIMES ###########
times <- load_times()

p <- ggplot(times, aes(x=runs, y=wild_mean, colour=factor(gradient))) + geom_line() +
 theme_classic() + labs(y="mean survival time", x="mutation and selection runs", title="Evolution of survival time in different gradient spaces") + 
  facet_wrap(.~factor(gradient), scales="free")

end_times <- sapply(gradients, function(i) get_end_times(i))
error <- sapply(gradients, function(i) get_end_var(i))


end_times <- data.frame(gradients, end_times, error)

end_times$min <- end_times$end_times - error
end_times$max <- end_times$end_times + error

write.csv(end_times, "survival_means.csv", col.names = TRUE)

p2 <- ggplot(end_times, aes(x=gradients, y=log(end_times))) + geom_point() + 
  theme_classic() + labs(x="Gradient values", y="final survival time mean", 
                         title = "Mean survival time of orbium evolved in different
                         gradients") + 
  geom_errorbar(aes(ymin=log(min), ymax=log(max))) 

print(p2)






