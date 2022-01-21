# CMEE Coursework, Week 7

This repository contains all of Izie's CMEE Coursework for Week7, Biological Computing in Python II. 

As guided by [TheMulQuaBio](https://mhasoba.github.io/TheMulQuaBio/notebooks), the files in this repository contain responses to practicals in chapter "Biological Computing in Python II".


## Languages
All code in this repository was written in Python 3.8.10, apart from one that is written in R 4.1.1.

## Dependencies 
1. **Numpy** - Tools for mathematical computing in Python ([linked here](https://numpy.org/))
2. **Scipy** - Algorithms for scientific computing in pyton ([linked here](https://scipy.org/))
3. **Matplotlib.pylab** - Data visualisation in Python ([linked here](https://matplotlib.org/))

## Structure and Usage
Respository contains three directories: 'code', 'data', 'results'. All scripts and code files are located in code. Appropriate files for which to run scripts on can be found in 'data'. All output files from scripts will be sent to 'results'

### Code
'Code' contains a series of Python3 scripts and programmes, which can be run from the bash command line, or from within iPython3.
- `LV1.py` - Runs and plots Lotka-Voltera model of Consumer-Resource population dynamics. Saves results to file.
```
%run LV1.py
```
- `profileme.py` - Simple script to illustrate profiling with. A series of functions that manipulate variables. 
```python
%run -p profileme.py  #-p for profiling
# Says how much time each function took to run
```
- `profileme2.py` - Another script for profiling- optimising `my_join` function and swapping a loop for a comprehension to help run faster.
```python
%run -p profileme2.py
```
-  `timeitme.py` - Script illustraiting use of timeit module in python. Makes timewise comparisons in console between functions imported from `profileme.py` and `profileme2.py`
```python
%run timeitme.py
%timeit my_squares_loops(iters)  # Returns system run time
```
- `MyFirstJupyterNB.ipynb` - Example jupyter notebook containing various practical examples.
```bash
jupyter notebook MyFirstJupyterNb.ipynb
```
- `TestR.R` - Very simple  R script for running from python workflow. Prints "Hello! This is R"
- `TestR.py` - Simple example of python workflow. Runs `TestR.R` from within python. Saves output of `TestR.R` and any error messages to results directory.
```
%run TestR.py
```
### Data
There are no data files needed to run the files in this repository.

### Results 
Results directory should be empty save a .gitignore - ignores everything in directory except for the .gitignore file. This is to satisfy minimum git requirement of a non-empty directory so our (effectly) empty results directory can be uploaded suit.

All scripts with output files should save outputs to results directory.

### Authors and Acknowledgement 
All files and practicals in this repository are directed by TheMulQuaBio. Special thanks to Dr. Alexander Christensen for leading us in Biological Computing in Python II. 

#### Author name and contact
 Izie Wood

 iw121@ic.ac.uk
