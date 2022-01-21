#!/usr/bin/Rscript --vanilla 
# Stats week exercises: Tuesday

#### 1. Calculate the statistical power of this test 

n <- 300 # number in each group
d <- 0.11 # effect size
a <- 0.044 #significance level

wp.t(n1 = n, n2 = n, d= d, alpha = a, type = "two.sample")
     
res.t <- wp.t(n1 = n, n2 = n, d= seq(0.05, 0.8, 0.05), alpha = a, type = "two.sample")
