#!/bin/bash
# Author: Izie Wood iw121@ic.ac.uk
# Script: variables.sh
# Desc:  Illustrates the use of various variables in bash
# Arguments: none
# Date: Oct 2021

# Special variables
echo "This script was called with $# parameters"  # $# is the number of parameters given
echo "The script's name is $0"  # argument in 0th place (ie. script name)

if [ $# == 0 ]; then  # If no arguments given..
echo "No arguments were given."
else
echo "The arguments are $@"
echo "The first argument is $1"
echo "The second argument is $2"
fi

# Assigned Variables; Explicit declaration
MY_VAR='some string'  #Bash variables always in uppercase, assigned without spaces between ==
echo # blank space
echo "The current value of the variable is" $MY_VAR
echo
echo 'Please enter a new string'
read MY_VAR # Take input from command line

if [ -z $MY_VAR ]; then  # If $MY_VAR empty..
echo "No string entered."
else
echo 'The current value of the variable is' $MY_VAR
fi
echo

## Assigned Variables; Reading multiple variables from user
echo "Enter two numbers separated by a space"
for (( ; ; )) # infinite loop to run over entries until correct values are given
do  
    read a b 
        if [ -z $a ]; then  # If a not given..
            echo 
            echo "No numbers entered"
            echo "Please enter two numbers separated by a space"
        elif [ -z $b ]; then # If only one number given...
            echo
            echo "Only one number entered"
            echo "Please enter your second numbers separated by a space"
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
