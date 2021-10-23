#!/usr/bin/env python3
# Filename: using_name.py
"""Script illustrating transfer of control when using 
if __name__ == '__main__' in a program"""

__appname__ = '[using_name.py]'
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

if __name__ == '__main__':
    print('This program is being run by itself')
else:
    print('I am being imported from another module')

print("This module's name is "+ __name__)