#!/usr/bin/env python3

"""Week 2 practical comprehension: Organise data into dictionary that maps
order names to sets of taxa"""
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

taxa = [ ('Myotis lucifugus','Chiroptera'),
         ('Gerbillus henleyi','Rodentia',),
         ('Peromyscus crinitus', 'Rodentia'),
         ('Mus domesticus', 'Rodentia'),
         ('Cleithrionomys rutilus', 'Rodentia'),
         ('Microgale dobsoni', 'Afrosoricida'),
         ('Microgale talazaci', 'Afrosoricida'),
         ('Lyacon pictus', 'Carnivora'),
         ('Arctocephalus gazella', 'Carnivora'),
         ('Canis lupus', 'Carnivora'),
        ]

# Write a short python script to populate a dictionary called taxa_dic 
# derived from  taxa so that it maps order names to sets of taxa.

taxa_dic = {} # create dictionary

# For each tuple, set order name as key in taxa_dic
# Set value for each dictionary key as a tuple of corresponding 
# taxa created via a list comprehension
for x in taxa:
        taxa_dic[x[1]] = set([y[0] for y in taxa if y[1] == x[1]])
        
# Print dictionary by taxa name 
for x in taxa_dic:
        print(x,":",taxa_dic[x])


# 
# An example output is:
#  
# 'Chiroptera' : set(['Myotis lucifugus']) ... etc.
#  OR,
# 'Chiroptera': {'Myotis lucifugus'} ... etc
