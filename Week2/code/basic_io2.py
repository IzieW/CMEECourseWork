# Author: Izie Wood (iw121@ic.ac.uk)
# Script: basic_io2.py
# Desc: Write values from list to text file
# Arguments: 1 -> text file
# Date: Oct 2021
#############################
# FILE OUTPUT
#############################
# save the elements of a list to a file
list_to_save = range(100)

f = open('../sandbox/testout.txt', 'w')
for i in list_to_save:
    f.write(str(i) + '\n') ## Add a new line at the end

f.close()