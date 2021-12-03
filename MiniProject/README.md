################## Mini Project ####################
This directory contains Izie Wood's CMEEMiniProject, which investigates the best model for fitting to empirical datasets of microbial growth records. 

### Languages
Scripts in this directory are mainly written in R 4.1.1, all marked .R
There is also one script in LaTeX and one in Bash

#### Dependencies
tidyverse #for data visualisation in R
minpack.lm #for plotting nonlinear equations using nlsLM()

##### Project Structure and Usage
Directory contains four subdirectories, "Code", "Results", "Graphics" and "Data".
To compile the project, run the script "../code/run_MiniProject.sh" from the terminal 

This will pull Data from Data directory, assemble models and charts, and compile a single LateX file containing the report

###### Author
Isabelle Wood
iw121@ic.ac.uk
