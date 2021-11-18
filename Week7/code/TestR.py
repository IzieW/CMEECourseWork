#!/usr/bin/env python3
__author__ = "Izie Wood (iw121@ic.ac.uk)"
"""Script to run TestR.R in python. Saves results to file"""

import subprocess

subprocess.Popen("Rscript --verbose TestR.R > ../results/TestR.Rout 2> ../results/TestR_errFile.Rout", shell =True).wait()