# !/bin/sh
# Author: Izie Wood iw121@ic.ac.uk
# Date: April 2022
# Desc: Compiles project proposal and cleans environment


# Compile script 
pdflatex proposal.tex
bibtex proposal
pdflatex proposal.tex
pdflatex proposal.tex
evince proposal.pdf & # open pdf

# Clean up directory
rm *.aux
rm *.log
rm *.bbl
rm *.blg

