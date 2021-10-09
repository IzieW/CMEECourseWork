#!/bin/bash
pdflatex $1
tr -d $1 ".*" | bibtex $1
pdflatex $1
pdflatex $1
tr -d $1 ".*" | evince $1.pdf &

## Cleanup
rm *.aux
rm *.log
rm *.bbl
rm *.blg 