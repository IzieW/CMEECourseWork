#!/usr/bin/bash
# Script: CompileLaTeX.sh
# Date: Jan 2022
# Desc: Compile a target LaTeX file into the current directory.
# Arguments: 1 -> path to the Tex file.

# If the .tex extension is in the argument, remove it.
without_ext=$(basename $1 .tex)

# 1. The first pdflatex run generates .log and .aux files (and an incomplete
#    .pdf).
# 		- All cite{…} arguments info that bibtex needs are written into the
#       .aux file.
# 2. The bibtex command results in bibtex reading the .aux file, producing
#    .bbl and .blg files.
# 		- bibtex takes the citation info in the .aux file and puts the relevant
#       entries into the .bbl file, formatted according to the bibliography style.
# 3. The second pdflatex run updates FirstExample.log and FirstExample.aux
#    (and a still-incomplete .pdf - the citations are not correctly formatted yet)
#     - The reference list in the .bbl generated above is included in the
#       document, and the correct labels for the in-text cite{...} commands 
#       are written in the .aux file .
# 4. The third and final pdflatex run then updates .log and .aux files one last
#    time, and now produces the complete .pdf file, with citations correctly formatted.
# 		- At this step, latex knows what the correct in-text citation labels are,
#       and includes them in the pdf document.
# 4. The final command opens the resulting pdf in evince, a utility for reading
#    pdf files.
pdflatex $without_ext.tex
bibtex $without_ext
pdflatex $without_ext.tex
pdflatex $without_ext.tex
evince $without_ext.pdf &

## Cleanup the auxiliary files used in the compilation procedure
rm *.aux
rm *.log
rm *.bbl
rm *.blg
