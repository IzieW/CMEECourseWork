### QUICK script looking to variance of data as runtimes increase
# Try to find best run time for time taken

require("tidyverse")


# Load data
run10 = read.csv("../mutate_and_select_all_10_runs.csv", header=TRUE)
run100 = read.csv("../mutate_and_select_all_100_runs.csv", header=TRUE)
run1000 = read.csv("../mutate_and_select_all_1000_runs.csv", header=TRUE)
run50 = read.csv("../mutate_and_select_all_50_runs.csv", header=TRUE)



# Look at wild type only
wild10 = run10$wild_time
wild100 = run100$wild_time
wild1000 = run1000$wild_time
wild50 = run50$wild_time
## Plot distributions
bell_curve <- function(data){
  m <- mean(data$wild_time)
  x <- seq(-150, 150, 1)
  y <- dnorm(x, mean=mean(data$wild_time), sd = sd(data$wild_time))
  
  plot(x, y, type="l")
  abline(v=m, col="blue")

}

par(mfrow=c(2,2))
plot(wild10, main="10 runs", type="l", xlab="run", ylab="survival time")
plot(wild50, main="50 runs", type="l", xlab="run", ylab="survival time")
plot(wild100, main="100 runs", type = "l", xlab="run", ylab="survival time")
plot(wild1000, main="1000 runs", type= "l", xlab="run", ylab="survival time")


plot(c(mean(wild10), mean(wild50), mean(wild100), mean(wild1000)), main="means", ylab = "means", xlab="runs")

f <- function(dat){
  m <- mean(dat)
  s <- sd(dat)

  return(s/m)
}

df <- data.frame(wild10, wild50, wild100, wild1000)

plot(c(f(wild10), f(wild50), f(wild100), f(wild1000)))
  

