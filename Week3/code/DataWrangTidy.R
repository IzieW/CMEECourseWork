#!/usr/bin/Rscript --vanilla
# Author: Izie Wood (iw121@ic.ac.uk)
# Desc: Script wrangles data using tidyverse
# Date: Oct 2021

###############################################
# Wrangle the Pound Hill Dataset in Tidyverse #
###############################################

########## Load tidyverse ###########
require(tidyverse)

############# Load the data set ###############
# use readr 
MyData <- readr::read_csv("../data/PoundHillData.csv", col_names = FALSE) # no headers

# header = true because we do have metadata headers
# read delim to auto detect separation values
MyMetaData <- readr::read_delim("../data/PoundHillMetaData.csv", col_names=TRUE)

############# Inspect the data set ###############
print("Loading data...")
print("Data:")
dplyr::glimpse(MyData)

print("Wrangling data..")

############# Transpose ###############
# To get those species into columns and treatments into rows 
MyData <- t(MyData)

############# Replace species absences with zeros ###############
MyData <- replace_na(MyData,"0")

############# Convert raw matrix to data frame ###############
TempData <- as.tibble(MyData[-1, ])
colnames(TempData) <- MyData[1,] # assign column names from original data

############# Convert from wide to long format  ###############

MyWrangledData <- tidyr::pivot_longer(TempData, 5:45, names_to = "Species", values_to = "Count")

MyWrangledData <- mutate(MyWrangledData, Cultivation = as.factor(Cultivation),
                         Block = as.factor(Block), Plot = as.factor(Plot), 
                         Quadrat = as.factor(Quadrat), Count = as.integer(Count))

print("Wrangled Data:")
dplyr::glimpse(MyWrangledData)
print(MyWrangledData)
