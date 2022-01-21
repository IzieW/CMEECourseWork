#### Exercises: Stats with Sparrows ####
# Oct 2021

# import data
d <- read.table("~/Downloads/SparrowSize.txt", header=TRUE)

### 1. How many repeats are there per bird per year? ###
d %>% group_by(Year) %>% count(BirdID)

### 2. How many individuals did we capture per year for each sex?
# Compute the numbers, devise a useful table format, and fill it in.

table(d$Sex.1,d$Year)

### 3. Think about how you can communicate (1) and (2) best in tables,
# and how you can visualize (1) and (2) using plots. 
#Produce several solutions and discuss in the groups the pros and cons
#of each. 

#(2) 

Sex <- d %>% group_by(Year) %>% count(Sex.1)



