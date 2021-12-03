#!/usr/bin/bash
# Author: Izie Wood (iw121@ic.ac.uk)
# Date : Nov 2021
# Desc: Script to run MiniProject workflow- trigger files to prepare data, 
# find starting values, model data, plot best fits, and compile a LaTeX report summarising the results
# Arguments <- none 

### 1 Prep Data
Rscript PrepData.R

### 2 Find starting values
Rscript FindStartingValues.R

### 3 Fit models 
Rscript ModelData.R

### 4 Plot models and analyse best fit 
Rscript AnalyseData.R 


######## Compile LaTeX script
 pdflatex FirstExample.tex
 bibtex MiniProject
 pdflatex FirstExample.tex
 pdflatex FirstExample.tex


