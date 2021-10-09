#!/bin/bash

# Shows the use of variables 
MyVar='some string'
echo "The current value of the variable is" $MyVar
echo 'Please enter a new string'
read MyVar
echo 'The current value of the variable is' $MyVar

## Reading multiple values
echo "Enter two numbers separated by space(s)"
read a b 
echo "You entered" $a "and" $b ". Their sum is:"
mysum=`expr $a + $b`
echo $mysum