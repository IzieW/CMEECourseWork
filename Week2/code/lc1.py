#!/usr/bin/env python3

"""Week 2 practical comprehension: Write list comprehsensions and for loops
to complete exercises on data set"""
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

birds = ( ('Passerculus sandwichensis','Savannah sparrow',18.7),
          ('Delichon urbica','House martin',19),
          ('Junco phaeonotus','Yellow-eyed junco',19.5),
          ('Junco hyemalis','Dark-eyed junco',19.6),
          ('Tachycineata bicolor','Tree swallow',20.2),
         )

#(1) Write three separate list comprehensions that create three different
# lists containing the latin names, common names and mean body masses for
# each species in birds, respectively. 

print("\nStep 1: list comprehensions\n")

print("Latin names:")
# List comprehension for latin names 
latin_names = [i[0] for i in birds]
print(latin_names, "\n")

print("Common names:")
# List comprehension for common names
common_names = [i[1] for i in birds]
print(common_names,"\n")

print("Body mass:")
# List comprehension for body mass
body_mass = [i[2] for i in birds]
print(body_mass, "\n")

# (2) Now do the same using conventional loops (you can choose to do this 
# before 1 !). 
print("Step 2: Conventional loops\n")

print("Latin names:")
# Conventional loop for latin names 
latin_names = []
for i in birds: 
    latin_names.append(i[0])
print(latin_names,"\n")

print("Common names:")
# Conventional loop for common names
common_names = []
for i in birds: 
    common_names.append(i[1])
print(common_names,"\n")

print("Body mass:")
# Conventional loop for body masses
body_mass = []
for i in birds: 
    body_mass.append(i[2])
print(body_mass)



# A nice example out out is:
# Step #1:
# Latin names:
# ['Passerculus sandwichensis', 'Delichon urbica', 'Junco phaeonotus', 'Junco hyemalis', 'Tachycineata bicolor']
# ... etc.
 