#! /usr/bin/bash
# Author: DashingDingos
# Script: run_get_TreeHeight.sh
# Description: Test get_TreeHeight.R and get_TreeHeight.py with trees.csv as your example file


red='\033[0;31m'
green='\033[0;32m'
nc='\033[0m'

echo Testing R TreeHeight script...
./get_TreeHeight.R ../data/trees.csv >/dev/null
echo R has finished running.

if [ ! -f ../results/trees_treeheights.csv ]
then
  echo -e "${red}FAILURE${nc} Output file does not exist."
else
  echo -e "${green}SUCCESS${nc}"
fi

echo
echo Testing Python TreeHeight script...
./get_TreeHeight.py ../data/trees.csv >/dev/null
echo Python has finished running.

if [ ! -f ../results/trees_treeheights_py.csv ]
then
  echo -e "${red}FAILURE${nc} Output file does not exist."
else
  echo -e "${green}SUCCESS${nc}"
fi
