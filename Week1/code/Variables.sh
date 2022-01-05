#!/bin/bash
# Author: Izie Wood iw121@ic.ac.uk
# Script: variables.sh
# Desc:  Illustrates the use of various variables in bash
# Arguments: none
# Date: Oct 2021

# Special variables

echo "This script was called with $# parameters"
echo "The script's name is $0"
echo "The arguments are $@"
echo "The first argument is $1"
echo "The second argument is $2"

# Assigned Variables; Explicit declaration
MY_VAR='some string'
echo # blank space
echo "The current value of the variable is" $MY_VAR
echo
echo 'Please enter a new string'
read MY_VAR # Take input from command line

if [ -z $MY_VAR ]; then # if no value entered, 
echo "No string entered."
fi
echo 'The current value of the variable is' $MY_VAR
echo

## Assigned Variables; Reading multiple variables from user
echo "Enter two numbers separated by space(s)"
for (( ; ; )) # infinite loop to run over entries until correct values are given
do  
    read a b 
        if [ -z $a ]; then
            echo 
            echo "No numbers entered"
            echo "Please enter two numbers separated by space(s)"
        elif [ -z $b ]; then
            echo
            echo "Only one number entered"
            echo "Please enter two numbers separated by space(s)"
        else
            echo "You entered" $a "and" $b. "Their sum is:"
            mysum=`expr $a + $b`
                echo $mysum
            
            ## Assigned Variables; Command substitution 
            MY_SUM=$(expr $a + $b)
            echo $MY_SUM
            exit
        fi
done
