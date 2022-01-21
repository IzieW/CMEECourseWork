#!/usr/bin/env python3

"""Week 2 practical comprehension: Organise data into dictionary that maps
order names to set of taxa"""

__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

taxa = [('Myotis lucifugus', 'Chiroptera'),
        ('Gerbillus henleyi', 'Rodentia',),
        ('Peromyscus crinitus', 'Rodentia'),
        ('Mus domesticus', 'Rodentia'),
        ('Cleithrionomys rutilus', 'Rodentia'),
        ('Microgale dobsoni', 'Afrosoricida'),
        ('Microgale talazaci', 'Afrosoricida'),
        ('Lyacon pictus', 'Carnivora'),
        ('Arctocephalus gazella', 'Carnivora'),
        ('Canis lupus', 'Carnivora'),
        ]


def make_dictionary(taxa):
    """Organise input list of taxa into dictionary of order names and sets of taxa"""
    taxa_dic = {}  # create empty dictionary
    for x in taxa:  # for each order name in list, populate set of taxa
        taxa_dic[x[1]] = set([y[0] for y in taxa if y[1] == x[1]]) # append all names that match dictionary key

    return taxa_dic


# Call above function and print dictionary line by line
dictionary = make_dictionary(taxa)
for x in dictionary:
    print(x, ":", dictionary[x])


# An example output is:
#  
# 'Chiroptera' : set(['Myotis lucifugus']) ... etc.
#  OR,
# 'Chiroptera': {'Myotis lucifugus'} ... etc
