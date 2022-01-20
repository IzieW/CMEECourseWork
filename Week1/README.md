# CMEE Coursework Week 1: 

This repository contains all of Izie's CMEE Coursework for Week1. 

As guided by [TheMulQuaBio](https://mhasoba.github.io/TheMulQuaBio/notebooks/01-Unix.html), the files in this repository contain responses to practicals in chapters "UNIX and LINUX" through to "Scientific Documents with LaTeX".

## Languages
All code in this repository was written in Bash 5.0.17, apart from one file, (FirstExample.tex), which is written in LaTeX (TexLive 2019).

## Dependencies
The following packages are required to run certain scripts in this repository:

1. **imagemagick**- Tools for manipulating digital images across a variety of formats ([Linked here](https://imagemagick.org/index.php))

## Structure and Usage
This repository's contents are  organised into three directories: 'code', 'data', 'results'. All scripts and code files are located in code. Appropriate files for which to run scripts on can be found in 'data'. All results will be sent to the results directory. 

### Code    
The code directory contains a series of scripts and files for inclass and assessed practicals occuring in Week1. 
- `boilerplate.sh` - Simple boilerplate script // Prints "This is a shells script!"
```bash
bash boilerplate.sh
```
- `variables.sh` - Illustrates special and assigned shell variables // Takes an optional two arguments from the command line and prints a series of statements about them
```bash
bash variables.sh [arg1] [arg2]
```
- `MyExampleScript.sh` - Illustrates use of environmental variables in bash // Prints "hello" to user
```bash
bash MyExampleScript.sh
```
- `tabtocsv.sh` - Takes input text file from command line and converts to csv  // saves to results directory
```bash
bash tabtocsv FILE.TXT
```
- `CountLines.sh` - Takes file from command line // Counts number of lines in file
```bash
bash CountLines.sh FILENAME
```
- `ConcatenateTwoFiles.sh` - Takes two files from command line // Concatenates files and saves new file to results directory
```bash
bash ConcatenateTwoFiles.sh FILE1 FILE2 [DESTINATION]  # Optionally can provide name of file to merge to
```
- `tiff2png.sh` - Converts all tiff files in working directory to png files  // Dependencies: imagemagik
```bash
bash tiff2png.sh
```
- `csvtospace.sh` - Takes input csv file and creates new space delimited text file in results
``` bash
bash csvtospace.sh FILE.CSV
```
- `UnixPrac1.txt` - Text file containing one-line UNIX shell commands that do a series of fasta exercises from TheMulQuaBio.
Each line can be copied and run from the terminal.
- `FirstExample.tex` - An example LaTeX file demonstrating LaTeX use
```bash
pdflatex FirstExample.tex
bibtex FirstExample
pdflatex FirstExample.tex
pdflatex FirstExample.tex
# or use CompileLaTeX below
bash CompileLaTex.sh FirstExample.tex
```
- `FirstBiblio.bib` - A bibliography to be compiled into above LaTeX file 
- `CompileLaTeX.sh` - A bash script that compiles input LaTeX file // Opens the compiled document and cleans working directory
``` bash
bash CompileLaTeX.sh SCRIPT.tex
# OR
bash CompileLaTeX.sh SCRIPT # with or without extension
```

### Data
- `fasta/` - Contains three different DNA sequences for use in `UnixPrac1.txt`

- `spawannxs.txt` - List of protected species pulled from web

- `Temperatures/` - CSV files containing temperatures for use in csvtospace.sh

### Results
Results directory should be empty save a `.gitignore` - ignores everything in directory except for the .gitignore file.
This is to satisfy minimum git requirement of a non-empty directory so our (effectly) empty results directory can be uploaded suit. 

All scripts with output files should save outputs to results directory. 

## Author name and contact
 Izie Wood

 iw121@ic.ac.uk