## Script to play out the stochastic attractant distribution
# as described in Godany, Khatri and Golstein 2017 paper on
# Chemotactic strategies. 

# Paper modelled various attractant distributions in a 1D virtual
# environment. 

require(tidyverse)

# Assign variables

T <- 10^3  # correlation time (how often changes happen)
L <- 100  # distance between peaks

l <- 100  # Length of world
p_max <- l/L  # Largest mode, p*
p <- seq(1, p_max, 0.1)  # All modes p to sum over
dt <- T/100  # Time increments when stochastic weights are updated

#### ONE DIMENSIONAL ATTRACTANT CONCENTRATIONS #####

get_Xp <- function(xp){
  # Stochastic weight. Return Xp(t+1) from input value of Xp(t),
  # ie. the previous value of Xp
  
    return( ( xp*(1-dt) ) + rnorm(1)*sqrt( (2*dt)/(T*p_max) ) )

}

get_Yp <- function(yp){
  # Stochastic weight. Return Xp(t+1) from input value of Xp(t),
  # ie. the previous value of Xp
    return( ( yp*(1-dt) ) + rnorm(1)*sqrt( (2*dt)/(T*p_max) ) )

}

zeta_p <- function(x, p){
  
  return( (2*pi*p*x)/l )
  
}


get_concentration_p <- function(p, x, xp, yp){
  # Get concentration at x for a given mode of p
  return ( xp*cos(zeta_p(x, p)) + yp*sin(zeta_p(x, p))  )
}

get_concentration_x <- function(x, xp, yp){
  # Return concentration at position x at time t. 
  # Using function as described by above paper
  # c(x, t) = max(0, sum all p:p*(Xp(t)*Cos(Zp(x)) + Yp(t)*Sin(Zp(x))))
  
  # Provide wit

  
  c <- max(0, sum(sapply(p, function(i) get_concentration_p(i, x, xp, yp))))
  
  return(c)  # Return concentration
  
}


# Set initial conditons 
Xp = 0
Yp = 0

get_concentrations_t <- function(positions, t, Xps, Yps){
  # Get concentrations of entire board x for one timestep t
  Cs <- sapply(1:l, function(i) get_concentration_x(positions[i],
                                                          Xps[i],
                                                    Yps[i]))
  return(Cs)
}

# Perform updates over all time steps T
positions <- rep(0, l) # Populate x of length l with zeros

concentrations <- matrix(NA, nrow=20, ncol=l)  # preallocate matrix
concentrations[1, ] <- get_concentrations_t(positions, t=1, Xps, Yps)


# Get Xps and Yps 
Xps <- matrix(0, nrow=T, ncol=l)
Yps <- matrix(0, nrow=T, ncol=l)

Xps[1, ] <- sapply(Xps[1, ], function(i) get_Xp(i))
Yps[1, ] <- sapply(Yps[1, ], function(i) get_Yp(i))
for (t in 2:T){
  if (t%%dt == 0){
    Xps[t, ] <- sapply(Xps[t-1,], function(i) get_Xp(i))
    Yps[t, ] <- sapply(Yps[t-1,], function(i) get_Yp(i))
  } else {
    Xps[t, ] <- Xps[t-1,]
    Yps[t, ] <- Yps[t-1,]
  }
  
}



for (t in 2:20){

  concentrations[t, ] <- get_concentrations_t(concentrations[t-1,], 
                                              t, 
                                              Xps[t,], 
                                              Yps[t,])
  
}







