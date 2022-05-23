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

##### TIMES #######
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

evolution_of_survival_time <- function(){
  # Get evolution of survival times for all gradients
p <- ggplot(times, aes(x=runs, y=wild_mean, colour=factor(gradient))) + geom_line() +
 theme_classic() + labs(y="mean survival time", x="mutation and selection runs", title="Evolution of survival time in different gradient spaces") + 
  facet_wrap(.~factor(gradient), scales="free")
print(p)
}

end_survival_times <- function(){
  # Get ending survival times and error bars for all gradients
end_times <- sapply(gradients, function(i) get_end_times(i))
error <- sapply(gradients, function(i) get_end_var(i))

end_times <- data.frame(gradients, end_times, error)

end_times$min <- end_times$end_times - error
end_times$max <- end_times$end_times + error

#write.csv(end_times, "survival_means.csv", col.names = TRUE)

p2 <- ggplot(end_times, aes(x=gradients, y=log(end_times))) + geom_point() + 
  theme_classic() + labs(x="Gradient values", y="final survival time mean", 
                         title = "Mean survival time of orbium evolved in different
                         gradients") + 
  geom_errorbar(aes(ymin=log(min), ymax=log(max))) 

print(p2)
}


######## THETA #############
load_data <- function(){
  theta <- read.csv("parameters/orbiumt4200n1r60_gradient0.5_parameters.csv", header=TRUE)

  
  for (i in gradients[2:10]){
    filename <- paste("parameters/orbiumt4200n1r60_gradient", i, "_parameters.csv", sep="")
    temp <- read.csv(filename, header=TRUE)
    temp <- as.data.frame(temp)
    theta <- bind_rows(theta, temp)
  }
  
  return(theta)
}

theta <- load_data()

plot_mutations <- function(){
   p <- ggplot(theta, aes(x=gradient, y = mutation)) + geom_line() + theme_classic() + 
     labs(x="gradient", y = "Total mutations", title="Total mutations in 1.5 hours of evolution in different gradients")
 
    print(p)
   }


plot_R <- function(){
  p <- ggplot(theta, aes(x = gradient, y = R)) + geom_line() + theme_classic() + 
    labs(x="gradient", y="R", title="R values evolved across different environmental gradients") + 
    geom_hline(yintercept=13, col="red", lty=3)
  
  print(p)
}

plot_T <- function(){
  p <- ggplot(theta, aes(x = gradient, y = T)) + geom_line() + theme_classic() + 
    labs(x="gradient", y="T", title="T values evolved across different environmental gradients") + 
    geom_hline(yintercept=10, col="red", lty=3)
  
  print(p)
}

plot_m <- function(){
  p <- ggplot(theta, aes(x = gradient, y = m)) + geom_line() + theme_classic() + 
    labs(x="gradient", y="m", title="mean growth values evolved across different environmental gradients") + 
    geom_hline(yintercept=0.15, col="red", lty=3)
  
  print(p)
}

plot_s <- function(){
  p <- ggplot(theta, aes(x = gradient, y = s)) + geom_line() + theme_classic() + 
    labs(x="gradient", y="s", title="std growth values evolved across different environmental gradients") + 
    geom_hline(yintercept=0.015, col="red", lty=3)
  
  print(p)
}

plot_b <- function(){
  p <- ggplot(theta, aes(x = gradient, y = b)) + geom_line() + theme_classic() + 
    labs(x="gradient", y="kernel peak (b)", title="kernel peak values evolved across different environmental gradients") + 
    geom_hline(yintercept=1, col="red", lty=3)
  
  print(p)
}







