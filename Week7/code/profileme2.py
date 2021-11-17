import numpy as np

def my_squares(iters):
    out = np.arange(1, iters)
    out = out ** 2
    return out

def my_join(iters, string):
    out = ''
    for i in range(iters):
        out += ", " + string
    return out

def run_my_func(x, y):
    print(x,y)
    my_squares(x)
    my_join(x,y)
    return 0

run_my_func(10000000, "My string")