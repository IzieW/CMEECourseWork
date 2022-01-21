#!/usr/bin/env python3 

"""Illustrates variable scope for global and local variables inside and out of functions"""
__author__ = 'Izie Wood (iw121@ic.ac.uk)'

# LOCAL VARIABLES: variables created inside functions are called "local" variables
# Their scope is limited to inside the function. 

## Example 1: Show scope of variables outside function
i = 1
x = 0
for i in range(10):
    x += 1
print(x)
print(i)  # "x" and "i" are last values in loop

## Example 2: The same variables, passed into a function...
# Return statement gives back values from function
i = 1  # Reset values
x = 0

def a_function(y):
    """Adds one to x for every number in input range"""
    x = 0
    for i in range(y):
        x += 1
    return x

# run function..
x = a_function(10) 
# both x and i are localised to the function above... 
print(x)  # x is now value returned from the function
print(i)  # "i" is still at its previous value, because it was not explicitly returned


# GLOBAL VARIABLES: you can designate variables to be "global" - visible inside and outside of functions

## Example 3: Global variables
_a_global = 10  # a global variable

if _a_global >= 5:
    _b_global = _a_global + 5  # also a global variable

print("Before calling a_function, outside the function, the value of _a_global is", _a_global)
print("Before calling a_function, outside the function, the value of _b_global is", _b_global)

def a_function():
    """Show varible scope in and out of function using global variables"""
    _a_global = 4  # a local variable

    if _a_global >= 4:
        _b_global = _a_global + 5  # also a local variable

    _a_local = 3

    print("Inside the function, the value of _a_global is", _a_global)
    print("Inside the function, the value of _b_global is", _b_global)
    print("Inside the function, the value of _a_local is", _a_local)

# call function..
a_function()

print("After calling a_function, outside the function, the value of _a_global is (still)", _a_global)
print("After calling a_function, outside the function, the value of _b_global is (still)", _b_global)

# print("After calling a_function, outside the function, the value of _a_local is ", a_local) : produces
# error because a_local not defined outside the function

## Example 4: All variables outside functions are also available inside functions
# Scope is assymetrical
_a_global = 10


def a_function():
    """shows scope of global varibale assigned outside of function, passed into function"""
    _a_local = 4

    print("Inside the function, the value _a_local is", _a_local)
    print("Inside the function, the value of _a_global is", _a_global)  #_a_global avaiable in function


a_function()

print("Outside the function, the value of _a_global is", _a_global)

# Example 5: Can assign global variables from inside a function using global keyword
_a_global = 10

print("Before calling a_function, outside the function, the value of _a_global is", _a_global)


def a_function():
    """Modifies global variables within function by using global keyword"""
    global _a_global
    _a_global = 5
    _a_local = 4

    print("Inside the function, the value of _a_global is", _a_global)
    print("Inside the function, the value _a_local is", _a_local)


a_function()

print("After calling a_function, outside the function, the value of _a_global now is", _a_global)


##  Example 6: nesting globals from within a function
def a_function():
    """Assigns global and function within function"""
    _a_global = 10

    def _a_function2():
        """Changes global"""
        global _a_global
        _a_global = 20

    print("Before calling a_function2, value of _a_global is", _a_global)

    _a_function2()

    print("After calling a_function2, value of _a_global is", _a_global) # Remains 10


a_function()

print("The value of a_global in main workspace / namespace now is", _a_global)  # Changes in main workspace to 20

## Compared to ...
## Example 7: Same as above but _a_global defined prior to function
_a_global = 10


def a_function():
    """Larger nested function"""

    def _a_function2():
        """Assign global"""
        global _a_global
        _a_global = 20

    print("Before calling a_function2, value of _a_global is", _a_global)

    _a_function2()

    print("After calling a_function2, value of _a_global is", _a_global) # Value inherited from a_function()
    # Modified everywhere from within _a_function2


a_function()

print("The value of a_global in main workspace / namespace is", _a_global)
