### File for testing Kernel equations used for Lenia Obstacle

sigmoid <- function(x){
  # Simple sigmoidal function
  return(1/(1+exp(-x)))
}

K_obstacle <- function(x){
  # Obstacle kernel as given by flowers lab
  # The lower the value given, the higher the kernel value returned
  # Ie. for distance matrices, the points closest to the cell will return the highest value
  return (exp((x/2)^2/2)*sigmoid(-10*(x/2 - 1)))
}

growth_obstacle <- function(x){
  
}
x <- seq(20, 0.1)
#plot(x, K_obstacle(x), type="l", color = "red")

plot(x, exp(-x), type="l")
lines(x, exp(-log(x)/2), col="red")