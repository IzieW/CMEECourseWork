#!/usr/bin/Rscript --vanilla 
# plot y and x with cartesian axis of given range

caxis <- function(c = 50){
  plot(x,y)
  segments(0, -(c), 0, c, lty = 3)
  segments(-(c), 0, c, 0, lty=3)
}