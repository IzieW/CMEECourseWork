# CMEE Coursework, Week 3

This repository contains all of Izie's CMEE Coursework for Week3, Biological Computing in R. 

As guided by [TheMulQuaBio](https://mhasoba.github.io/TheMulQuaBio/notebooks/), the files in this repository contain responses to practicals in from chapters "Biological Computing in R" and "Data Management and Visualisation".

## Languages
The code in this repository was written in R 4.1.1, LaTeX (TexLiv 2019), Bash 5.0.17, and one file in Python 3.8.10.

## Dependencies 
1. **Tidyverse** - R packages for data science. Help with data manipulation and visualisation ([linked here](https://www.tidyverse.org/))
2. **maps** - Package for drawing geographical maps ([linked here](https://cran.r-project.org/web/packages/maps/maps.pdf))

## Structure and Usage
Respository contains three directories: 'code', 'data', 'results'. All scripts and code files are located in code. Appropriate files for which to run scripts on can be found in 'data'. All script outputs will be saved to 'results'

### Code
Code contains a series of R scripts, LaTeX, Bash and one python script, which can be run from the bash command line, or from within R.
- `basic_io.R` - Script illustrating basic import and exports in R. 
```
source("basic_io.R")
```
- `control_flow.R` - Script illustrating use of various control flows in R. Prints a series of variables as lists and ranges are looped over. 
```
source("control_flow.R")
```
- `break.R` - Script illustrating use fo break functions in loops. Breaks out of loop after 10 iterations. 
```
source("break.R")
```
- `next.R` - Script illustrating use of next in loop to pass to next iteration of loop.
```
source("next.R")
```
- `boilerplate.R` - Boilerplate script illustrating use of functions and arguments in R. Prints names of arguments and their variable type for example functions when called. 
```
source("boilerplate.R")
```
- `R_conditonals.R` - Demonstrates use of conditionals in R. Several functions which produce statements about input variables based on conditions in function. When sourced, will print a series of test runs.
```
source("R_conditionals.R")
```
- `TreeHeight.R` - Loads data from ../data/trees.csv,  calculates tree heights and writes results to ../results/TreeHts.csv
```
source("TreeHeight.R")
```
-`get_TreeHeight.R` - Group Work practical. Same as `TreeHeight.R` except this script takes optional input files from command line and writes to file whose name includes the input filename.
```
Rscript get_TreeHeight.R [FILE.CSV]
```
-`get_TreeHeight.py` - Group Work practical. Script does the same thing as `TreeHeight.R` but is written in python3. 
```
python3 get_TreeHeight.py [FILE.CSV]
```
- `run_get_TreeHeight.sh` - Bash script that runs the above two scripts. 
```
bash run_get_TreeHeight.sh
```
- `Vectorize1.R` - Sums elements using loops and using vectorised sum() function. Prints system run time for each. 
```
source("Vectorize1.R")

```
- `preallocate.R` - Demonstrates memory allocation in R when using pre-allocated and non pre-allocated vectors. Prints system run time for each. Prints memory allocation after each loop.
```
source("preallocate.R")
```
- `apply1.R` - Illustrates use of apply function in R. Prints results to variety of matrix manipulations. 
```
source("apply1.R")
```
- `apply2.R` - Illustrates using apply for functions of your own definition. Prints results to function that multiplies its input values by 100. 
```
source("apply2.R")
```
- `sample.R` - Script illustrating a variety of vectorization methods. Prints the system run time for each one. 
```
source("sample.R")
```
- `Vectorize2.R` - Week 3 practical. Provides better vectorization of the Stochastic Ricker model. Prints system run time for both functions. 
```
source("Vectorize2.R")
```
- `browse.R` - Script demonstrates browser() function in R. Plots exponential growth over 10 generations. 
```
source("browse.R")
```
- `try.R` - Script that demonstrates use of try() keyword in R. 
```
source("try.R")
```
- `Florida_Warming.R` - Performs permutation analysis over data on Florida Temperatures to see Temperatures are rising over time. Saves results to ../results/permutation_results.pdf
```
source("Florida_Warming.R")
```
- `Florda_Warming.Tex` - LaTeX script presents findings from above permutation analysis in write up. 
```bash
bash Florida_Warming.sh  # Compile document
```
- `Florida_Warming.sh` - Bash script that runs `Florida_Warming.R`, compiles and opens the LaTeX `Florida_Warming.Tex` report, and cleans up the work space.
```
bash Florida_Warming.sh
```
- `TAutoCorr.R` -  Group Work practical. Analyses autocorrelation between Florida Temperatures each year to see if temperatures of one year are significantly correlated with the next. Plots and saves results for writeup. 
```
source("TAutoCorr.R")
```
- `TAutoCorr.tex` - Group Work practical. LaTeX script presenting the results found from above `TAutoCorr.R` script. 
```bash
bash CompileLaTeX.sh TAutoCorr.tex
```
- `CompileLaTeX.sh` - Bash script that compiles and opens LaTeX document.
```
bash CompileLaTeX.sh
```
- `DataWrang.R` - Script loads data from file and demonstrates series of data wrangling steps. 
```
source("DataWrang.R")
```
- `DataWrangTidy.R` - Completes the same data wrangling steps as `DataWrang.R` but using tools from tidyverse.
```
source("DataWrangTidy.R")
```
- `PP_Dists` - Plots Distribution of Predator, Prey Mass, and Predator/Prey mass by feeding interaction in three charts. Saves charts to pdf, as well as csv file containing mean, median log predator and prey mass, and the predator-prey size ratio by feeding type. 
``` 
source("PP_Dists.R")
```
- `Girko.R` - Plots Girko's law simulation and saves results to ../results/Girko.pdf
```
source("Girko.R")
```
- `MyBars.R` - Plots histogram with annotations. Saves results to ../results/MyBars.R
```
source("MyBars.R")
```
- `plotLin.R` - Plots and annotates regression line with mathematical notation on axis. Saves results to MyLinReg.pdf
```
source("plotLin.R")
```
- `PP_Regress.R` - Plots given figure and saves to pdf. Also saves csv file containing regression results of fitted lines. 
``` 
source("PP_Regress.R")
```
- `GPDD_Data.R` - Loads GPDD data and creates world map with all locations from data superimposed onto it. 
```
source("GPDD_Data.R")
```
### Data
- `EcolArchives-EO89-51-D1.csv` - Dataset on Consumer-Resource body mass ratios. Taken from [figshare](https://figshare.com/collections/PREDATOR_AND_PREY_BODY_SIZES_IN_MARINE_FOOD_WEBS/3300257)
- `KeyWestAnnualMeanTemperature.RData` - Annual temperatures in Key West Florida, USA over the last century.
- `PoundHillData.csv` - Dataset colleted by students in past Silwood field course
- `PoundHillMetaData.csv` - Metadata explaining variables in `PoundHillData.csv`
- `trees.csv` - Dataset of different tree distances from ground and their angles. 
- `GPDDFiltered.RData` - Dataset from the Global Population Dynamics Database
- `Results.txt` - Dataset for use in `MyBars.R`

### Results
Results directory should be empty save a .gitignore - ignores everything in directory except for the .gitignore file. This is to satisfy minimum git requirement of a non-empty directory so our (effectly) empty results directory can be uploaded suit.

All scripts with output files should save outputs to results directory.

## Authors and Acknowledgement
All files and practical in this repository are directed by TheMulQuaBio. Special thanks to Dr.Samraat Pawar for leading the work in this repository. 

All group work files (`get_TreeHeight.R`, `get_TreeHeight.py`, `run_get_TreeHeight.sh`, `TAutoCorr.R`, `TAutoCorr.tex` and `CompileLaTeX.sh`) were made collaboratively by members of the Dashing Dingos. An amazing team to work with. 

## Author name and contact
 Izie Wood
 
 iw121@ic.ac.uk
