#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood (iw121@ic.ac.uk)
# Desc: Script plots Girko's law simulation 
# Arguments: none
# Date: Nov 2021

require(tidyverse)
####### Function to calculate the ellipse ########
build_ellipse <- function(hradius, vradius){
  npoints = 250
  a <- seq(0, 2 * pi, length = npoints + 1)
  x <- hradius * cos(a)
  y <- vradius * sin(a)
  return(data.frame(x = x, y = y)) # returns ellipse
}

####### Build matrix and find values #########
N <- 250 # Assign size of matrix 

M <- matrix(rnorm(N * N), N, N) # Build matrix 

eigvals <- eigen(M)$values #Find the eigenvalues

eigDF <- data.frame("Real" = Re(eigvals), "Imaginary" = Im(eigvals)) # build a dataframe

my_radius <- sqrt(N)

ellDF <- build_ellipse(my_radius, my_radius) # Data frame to plot ellipse

names(ellDF) <- c("Real", "Imaginary") # rename columns

################ Plot ellipse #################
# Plot the eigenvalues
p <- ggplot(eigDF, aes(x = Real, y = Imaginary))
p <- p + 
  geom_point(shape = I(3)) + 
  theme(legend.position = "none")

# now add the vertical and horizontal line
p <- p + geom_hline(aes(yintercept = 0))
p <- p + geom_vline(aes(xintercept = 0))

# finally, add the ellipse 
p <- p + geom_polygon(data = ellDF, aes(x = Real, y = Imaginary,
                                        alpha = 1/20, fill = "red"))
p
############# save plot to pdf ##############
pdf("../results/Girko.pdf")
print(p)
dev.off() # close file
print("Done! File saved to ../results/Girko.pdf")
