# CMEE Coursework, Week 2

This repository contains all of Izie's CMEE Coursework for Week 2: Biological Computing in Python 1. 

As guided by [TheMulQuaBio](https://mhasoba.github.io/TheMulQuaBio/notebooks/), the files in this repository contain responses to practicals in chapter "Biological Computing in Python I"

## Languages
All code in this repository was written in Python 3.8.10.

## Structure and Usage 
This respository contains three directories: 'code', 'data', 'results'. All scripts and code files are located in code. Appropriate files for which to run scripts on can be found in 'data', and all output script files will be saved to 'results'

## Dependencies
No extraneous packages or modules are needed to run the scripts in this repository.

### Code
The code repository contains a series of Python3 scripts and programmes for independent and group work practicals in Week2. They can all be run from the bash command line, or from within iPython3.

- `loops.py` - Script illustrating use of loops in python. Prints a series of variables spun from for and while loops
```
%run loops.py
```
- `MyExampleScript.py` - Script demonstrating functions in python. Function takes input value and returns squared value.
```
%run MyExampleScript.py
```
- `cfexercises1.py` - Demonstrates use of control flows in python. Contains aseries of functions that manipulate input variables in various ways, and prints results to terminal. 
```
%run cfexercises.py
```
- `cfexercises2.py` - Script containing more functions that illustrate use of control flows and loops in python. Prints a series of test output values from functions. 
```
%run cfexercises2.py
```
- `oaks.py` - Demonstrates use of for loops versus list comprehensions in python. Finds taxa that are oak trees from a list fo species using both for loops and comprehensions. Prints results. 
```
%run oaks.py
```
- `scope.py` - Illustrates global and local variable scope inside and out of functions in python. Prints a series of statements about the values of variables as they are passed in and out of different functions. 
```
%run scope.py
```
- `basic_io1.py` - Demonstrates how to import lists from files. Imports data from ../data/test.txt and prints contents to terminal- first as normal, then removing blank spaces.
```
%run basic_io1.py
```
- `basic_io2.py` - Demonstrates how to write lists to files. Exports contents of a list to ../results/testout.txt. 
```
%run basic_io2.py
```
- `baisc_io3.py` - Demonstrates importing and exporting objects using pickle. Writes dictionary to ../results/testp.p, reloads object to dictionary, and prints dictionary.
```
%run basic_io3.py
```
- `basic_csv.py` - Demonstrates how to read and write data from csv files. Imports data from .../data/testcsv.csv and prints lines to terminal, exports to file containing only species name and body mass.
```
%run basic_csv.py
```
- `boilerplate.py` - Simple boilerplate template program. Prints "This is a boilerplate" when called from command line. 
```
%run boilerplate.py
```
- `using_name.py` - Script demonstrates tranfer of control when using `if __name__ = '__main__' in a program. Run program from command line, and import it to see how name changes based on how program is called. 
```python
%run using_name.py
import using_name.py  # output changes
```
- `sysargv.py` - Demonstrates use of sys.argv in python. Takes any number of optional arguments from command line. Prints script name, number of arguments, and the arguments given in a string. 
```
%run sysargv.py [argi]
```
- `control_flow.py` - Example program that uses various control flow functions. When called from command line,
program runs all functions with various test values and prints results. 
```
%run control_flow.py
```
- `lc1.py` - Week2 practical. Creates lists of latin names, common names, and body mass for each bird in tuple of tuples, first using list comprehensions, then for loops. Prints results to terminal. 
```
%run lc1.py
```
- `lc2.py` - Week 2 practical. Creates month/rainfall tuples for months where rainfall was greater than 100mm and less than 50mmm, first using list comprehensions, then for loops. Prints results to terminal. 
```
%run lc2.py
```
- `dictionary.py` - Week 2 practical. Organises species names into dictionary by their taxa. Prints dictionary line by line. 
```
%run dictionary.py
```
- `tuple.py` - Week 2 practical. For each species in tuple of tuples of birds, prints latin name, common name and body mass on a separate line.
```
%run tuple.py
```
- `test_control_flow.py` - Illustrates use of doctests for testing functions in program. Same functionality as `control_flow.py` program above, but with doctests.
```python
%run test_control_flow.py  # Run as normal
%run test_control_flow.py -v  # Run with doctests
```
- `debugme.py` - Demonstrates use of try-except key words in function to catch anticipated errors without stopping function.
```
%run debugme,py
```
- `align_seqs.py` - Week 2 practical. Loads DNA sequences from ../data/Example_seqs.csv and finds alignment that maximises base matches. Saves best alignment and the number of matches ("best score") to ../results/align_seqs.txt.
```
%run align_seqs.py
```
- `align_seqs_fasta.py` - Group Work practical. Same as `align_seqs.py` except for it will take any two fasta sequences (in separate files) as input from command line. If no files are given, program will use default files. Results are written to ../results/align_fasta.txt
```
%run align_seqs_fasta.py [SEQ1.FASTA] [SEQ2.FASTA]
```
- `align_seqs_better.py` - Group Work practical. Same as `align_seqs_fasta.py` except the program will save all possible alignments that share the same best score. Results are written to ../results/align_better.csv
```
%run align_seqs_better.py [SEQ1.FASTA] [SEQ2.FASTA]
```
- `oaks_debugme.py` - Modified version of `oaks.py`, this program is fitted with doctests, and finds oaks based on a similarity function, such that it can flag possible typos. Saves list of oaks to ../results/JustOaksData.csv
```python
%run oaks_debugme.py  # Run as normal
%run oaks_debugme.py -v  # Run with doctests
```
- `group_oaks_debugme.py` - Group work practical. Finds oaks in a list of species and saves them to ../results/JustOaksData.csv under appropriate headers. 
```
%run group_oaks_debugme.py
```
### Data
- `Example_seqs.csv` - Example DNA sequences to run on `align_seqs.py`
- `fasta/` - directory containing three fasta files of DNA sequences
- `JustOaksData.csv` - List of trees and species names for use in `oaks.py` `oaks_debugme.py` and `group_oaks_debugme.py`
- `testcsv.csv` - csv for manipulation in `basic_csv.py`
- `test.txt` - text file for manipulation in `basic_io1.py`

### Results
Results directory should be empty save a `.gitignore` - ignores everything in directory except for the .gitignore file. This is to satisfy minimum git requirement of a non-empty directory so our (effectly) empty results directory can be uploaded suit.

All scripts with output files should save outputs to results directory.

## Authors and Acknowledgement 
All files and practicals in this repository are directed by TheMulQuaBio. Special thanks to Dr.Samraat Pawar for leading the work in this repository.

All group work files (`align_seqs_fasta.py` `align_seqs_better.py`, and `group_oaks_debugme.py`) were made collaboratively by members of the DashingDingos group. An amazing team to work with. 

## Author name and contact
 Izie Wood
 iw121@ic.ac.uk
