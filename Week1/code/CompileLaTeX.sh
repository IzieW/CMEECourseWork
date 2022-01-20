#!/usr/bin/bash
# Author: Izie Wood (iw121@ic.ac.uk)
# Desc: Script compiles input LaTeX document, cleans workspace and opens document
# Arguments: 1 <- LaTeX file 
# Date: Oct 2021

# Check for valid arguments
if [ -z $1 ]; then # feedback if no argument is given, throw error
    echo "Error: Missing file to compile. Please give name of LaTeX file after the command"
    echo "ex: CompileLaTeX.sh FILE"
    exit 2
elif [ ! -e $1 && ! -e $1.tex ]; then # if file doesn't exist (with or without extension)
    echo "Error in $1: no such file or directory"
    exit 2
fi

#Compile document
if [[ $1 == *".tex" ]]; then # If argument ends in .tex
set -- ${1::-4} # If so, use set to edit argument, and parameter expansion to shave last four characters
fi # if file given without .tex, no transformations needed

pdflatex $1.tex # Generates two files *.log and *.aux, and incomplete *.pdf. Bibtex reads .aux file
bibtex $1 # Bibtex reads .aux file, produces *.bbl and *.blg, taking relevant info from .aux and putting into .bbl
pdflatex $1.tex # Updates .log, .aux and .pdf, reference list appears in .pdf, but not in-text references
pdflatex $1.tex # Runs .aux and .log one more time producing the complete .pdf
evince $1.pdf & # Opens the pdf 

## Cleanup
rm *.aux
rm *.log
rm *.bbl
rm *.blg 

echo -e "\nDone! Compiled script saved to $1.tex"

exit 0  #Programme run successfully
