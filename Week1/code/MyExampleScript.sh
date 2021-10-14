#!/bin/bash
# Author: Izie Wood iw121@ic.ac.uk
# Script: MyExampleScript.sh
# Desc: Demonstrate use of variables through explicit declaration, two ways of saying Hello $USER
# Arguments: None
# Date: Oct 2021

msg1="Hello"
msg2=$USER
echo "$msg1 $msg2"
echo "Hello $USER"
echo