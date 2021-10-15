#!/bin/bash
if [ -z $1 ]; then # feedback if no argument is given. Exit.
    echo "Error- Please give name of LaTeX file after the command"
    echo "ex: CompileLaTeX.sh FILE"
    echo "Try again"
    exit
fi
if [[ $1 == *".tex" ]]; then # Check if argument given ends in .tex
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