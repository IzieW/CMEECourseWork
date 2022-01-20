#!/usr/bin/bash
# Author: Izie Wood (iw121@ic.ac.uk)
# Date: Nov 2021
# Desc: Compile Florida_Warming.tex Latex report

pdflatex Florida_Warming.tex
pdflatex Florida_Warming.tex
pdflatex Florida_Warming.tex
evince Florida_Warming.pdf &

## Cleanup 
rm *.aux
rm *.log
rm *.bbl
rm *.blg