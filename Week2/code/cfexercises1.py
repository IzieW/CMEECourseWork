# What does each of foo_x do?
def foo_1(x): # Returns x to the power of 0.05
    return x **0.5 

def foo_2(x, y): # Returns the greater variable
    if x > y:
        return x
    return y

def foo_3(x, y, z): # Reorders x and y according to which is bigger, then reorders y and x according to which is bigger.
    if x > y:
        tmp = y
        y = x
        x = tmp
    if y > z:
        tmp = z
        z = y
        y = tmp
    return [x, y, z]

def foo_4(x): # calculates the factorial of x
    result = 1
    for i in range(1, x + 1):
        result = result * i
    return result

def foo_5(x): # a recursive function that calculates the factorial of x 
    if x == 1:
        return 1
    return x * foo_5(x - 1)

def foo_6(x): # Calculate the factorial of x in a different way
    facto = 1
    while x >= 1:
        facto = facto * x
        x = x -1
    return facto