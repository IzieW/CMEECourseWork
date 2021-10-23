#!/usr/bin/env python3

"""Week 2 practical comprehension: Using for loop print 
each value in tuple"""
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

birds = ( ('Passerculus sandwichensis','Savannah sparrow',18.7),
          ('Delichon urbica','House martin',19),
          ('Junco phaeonotus','Yellow-eyed junco',19.5),
          ('Junco hyemalis','Dark-eyed junco',19.6),
          ('Tachycineata bicolor','Tree swallow',20.2),
        )

# Birds is a tuple of tuples of length three: latin name, common name, mass.
# write a (short) script to print these on a separate line or output block by species 

# For loop to parse through tuples, printing each value
for x in birds: 
    print("\nLatin name:",x[0])
    print("Common name:",x[1])
    print("Body mass:",x[2])


# A nice example output is:
# 
# Latin name: Passerculus sandwichensis
# Common name: Savannah sparrow
# Mass: 18.7
# ... etc.

# Hints: use the "print" command! You can use list comprehensions!