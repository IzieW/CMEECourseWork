#!/bin/bash
# Author: Izie Wood iw121@ic.ac.uk
# Script: MyExampleScript.sh
# Desc: Illustrates use of environmental variables
# Arguments: None
# Date: Oct 2021

msg1="Hello"
msg2=$USER # Take name of user, an environmental variable
echo "$msg1 $msg2"
echo "Hello $USER"
echo
