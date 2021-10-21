# Average UK Rainfall (mm) for 1910 by month
# http://www.metoffice.gov.uk/climate/uk/datasets
rainfall = (('JAN',111.4),
            ('FEB',126.1),
            ('MAR', 49.9),
            ('APR', 95.3),
            ('MAY', 71.8),
            ('JUN', 70.2),
            ('JUL', 97.1),
            ('AUG',140.2),
            ('SEP', 27.0),
            ('OCT', 89.4),
            ('NOV',128.4),
            ('DEC',142.2),
           )

# (1) Use a list comprehension to create a list of month,rainfall tuples where
# the amount of rain was greater than 100 mm.
print("\nStep 1: List comprehensions\n")
print("Months and rainfall where rainfall was greater than 100mm:")
rainfall_100mm = [i for i in rainfall if i[1] > 100] # if after for since no else
print(rainfall_100mm)
 
# (2) Use a list comprehension to create a list of just month names where the
# amount of rain was less than 50 mm. 
print("\nStep 2: List comprehensions\n")
print("Months where rainfall was less than 50mm:")
months_50mm = [i[0] for i in rainfall if i[1] < 50]
print(months_50mm)

# (3) Now do (1) and (2) using conventional loops (you can choose to do 
# this before 1 and 2 !). 
print("\nStep 3: Conventional loops\n")

# Months + rainfall where rainfall was greater than 100mm
print("Months and rainfall where rainfall was greater than 100mm:")
rainfall_100mm = []
for i in rainfall: 
    if i[1] > 100:
        rainfall_100mm.append(i)

print(rainfall_100mm)

print("\nMonths where rainfall was less than 50mm")
months_50mm = []
for i in rainfall:
    if i[1] < 50:
        months_50mm.append(i[0])

print(months_50mm)

# A good example output is:
#
# Step #1:
# Months and rainfall values when the amount of rain was greater than 100mm:
# [('JAN', 111.4), ('FEB', 126.1), ('AUG', 140.2), ('NOV', 128.4), ('DEC', 142.2)]
# ... etc.