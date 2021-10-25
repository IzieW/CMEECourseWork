#!/usr/bin/env python3 

"""Scripts and functions illustrating scope of variables"""
__author__ = 'Izie Wood (iw121@ic.ac.uk)'

## first example ##
i = 1
x = 0
for i in range(10):
    x += 1
print(x)
print(i)

## make the above a function ##
i = 1
x = 0
def a_function(y):
    """Variable scope excersise in function"""
    x = 0
    for i in range(y):
        x += 1
    return x
x = a_function(10)
print(x)
print(i)

## designating global functions ##
_a_global = 10 # a global variable

if _a_global >= 5:
    _b_global = _a_global + 5 # also a global variable
    
print("Before calling a_function, outside the function, the value of _a_global is", _a_global)
print("Before calling a_function, outside the function, the value of _b_global is", _b_global)

def a_function():
    """Show varible scope in and out of function using global variables"""
    _a_global = 4 # a local variable
    
    if _a_global >= 4:
        _b_global = _a_global + 5 # also a local variable
    
    _a_local = 3
    
    print("Inside the function, the value of _a_global is", _a_global)
    print("Inside the function, the value of _b_global is", _b_global)
    print("Inside the function, the value of _a_local is", _a_local)
    
a_function()

print("After calling a_function, outside the function, the value of _a_global is (still)", _a_global)
print("After calling a_function, outside the function, the value of _b_global is (still)", _b_global)

#print("After calling a_function, outside the function, the value of _a_local is ", a_local)

## another global function ##
_a_global = 10

def a_function():
    """shows global function assigned outside of function"""
    _a_local = 4
    
    print("Inside the function, the value _a_local is", _a_local)
    print("Inside the function, the value of _a_global is", _a_global)
    
a_function()

print("Outside the function, the value of _a_global is", _a_global)

## modify the global within a function ##_a_global = 10

print("Before calling a_function, outside the function, the value of _a_global is", _a_global)

def a_function():
    """Example of modifying global within a function"""
    global _a_global
    _a_global = 5
    _a_local = 4
    
    print("Inside the function, the value of _a_global is", _a_global)
    print("Inside the function, the value _a_local is", _a_local)
    
a_function()

print("After calling a_function, outside the function, the value of _a_global now is", _a_global)
 
## nesting globals from within a function ##
def a_function():
    """Assigns global"""
    _a_global = 10

    def _a_function2():
        """Changes global"""
        global _a_global
        _a_global = 20
    
    print("Before calling a_function2, value of _a_global is", _a_global)

    _a_function2()
    
    print("After calling a_function2, value of _a_global is", _a_global)
    
a_function()

print("The value of a_global in main workspace / namespace now is", _a_global)

## Compared to ... ##
_a_global = 10

def a_function():
    """Arbitrary function"""

    def _a_function2():
        """Assign global"""
        global _a_global
        _a_global = 20
    
    print("Before calling a_function2, value of _a_global is", _a_global)

    _a_function2()
    
    print("After calling a_function2, value of _a_global is", _a_global)

a_function()

print("The value of a_global in main workspace / namespace is", _a_global)
