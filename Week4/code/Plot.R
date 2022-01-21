plot3 <- function(a){
  for (i in a){
    plot(x, i, xlim=c(-10,10), ylim=c(-20, 20), col="red", pch=19, cex=0.8, main
         =paste("Cov=",round(cov(x,y1),digits=2)," Cor=",round(cor(x,i),digits=2)))
  }
}