#!/usr/bin/Rscript --vanilla 
# Author: Izie Wood
# Desc: Script to evaluate genetic diversity of brown bear 
# population 
# Date: Nov 2021

##### Packages ####
require(tidyverse)

############## Load Data ###############
bears <- read.csv("../data/bears.csv", header = TRUE, 
                  stringsAsFactors =  FALSE, colClasses = rep("character", 10000))

######## Identify which positions are SNPs #########
is.SNP <- function(i){ # function identifies cols with SNP
      if (length(unique(i)) > 1){
        return(TRUE)
      }else{
        return(FALSE)
      }
}

SNP <- select(bears, where(function(i) is.SNP(i))) # saves columns with SNP to DF

##### Calculate, print and visualize allele frequencies ##### 
# for each SNP 
sumdf <- function(x){
  
}