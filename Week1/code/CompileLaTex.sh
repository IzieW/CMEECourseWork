#!/bin/bash
pdflatex $1 # Generates two files *.log and *.aux, and incomplete *.pdf. Bibtex reads .aux file
bibtex $1 # Bibtex reads .aux file, produces *.bbl and *.blg, taking relevant info from .aux and putting into .bbl
pdflatex $1 # Updates .log, .aux and .pdf, reference list appears in .pdf, but not in-text references
pdflatex $1 # Runs .aux and .log one more time producing the complete .pdf
evince $1.pdf & # Opens the pdf 

## Cleanup
rm *.aux
rm *.log
rm *.bbl
rm *.blg 