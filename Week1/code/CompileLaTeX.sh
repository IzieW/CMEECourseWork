#!/bin/bash
pdflatex $1.tex | tr -s
bibtex $1 | tr -d ".tex"
pdflatex $1.tex | tr -s
pdflatex $1.tex | tr -s
evince $1.pdf & #evince opens document to view

## Cleanup
rm *.aux
rm *.log
rm *.bbl
rm *.blg