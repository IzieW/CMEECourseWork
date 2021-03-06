#!/usr/bin/env python3

"""
Example of python program that uses various control flows. Contains a series of functions 
that manipulate variables in given ways. When called from command line, program runs functions
with various test values and prints results.
"""
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

## imports ##
import sys  # module to interface our programmes with the operating system

## functions ##
def even_or_odd(x=0):  # if not specified, x should take value 0.

    """Find whether an input number is even or odd."""
    if x % 2 == 0:  # If input is divisble by 2...
        return "%d is Even!" % x  #%d used to format returned value as decimal
    return "%d is Odd!" % x


def largest_divisor_five(x=120):  #default to 120
    """Find which is the largest divisor of input value amoung 2, 3, 4, 5."""
    largest = 0
    if x % 5 == 0:
        largest = 5
    elif x % 4 == 0:  # "else if"
        largest = 4
    elif x % 3 == 0:
        largest = 3
    elif x % 2 == 0:
        largest = 3
    else:  # When all other (if, elif) conditions are not met
        return "No divisor found for %d!" % x  # Each function can return a value or variable.
    return "The largest divisor of %d is %d" % (x, largest)


def is_prime(x=70):
    """Find whether input number is prime."""
    for i in range(2, x):  # "range" returns a sequence of integers
        if x % i == 0:
            print("%d is not a prime: %d is a divisor" % (x, i))
            return False
    print("%d is a prime!" % x)
    return True


def find_all_primes(x=22):
    """Find all the primes numbers within input range"""
    allprimes = []
    for i in range(2, x + 1):
        if is_prime(i):
            allprimes.append(i)
    print("There are %d primes between 2 and %d" % (len(allprimes), x))
    return allprimes


def main(argv):
    """Main entry points of argument.
    Call above functions with test inputs, and print results."""
    print(even_or_odd(22))
    print(even_or_odd(33))
    print(largest_divisor_five(120))
    print(largest_divisor_five(121))
    print(is_prime(60))
    print(is_prime(59))
    print(find_all_primes(100))
    return 0


if __name__ == "__main__":
    # Run main function when called from command line
    status = main(sys.argv)
    sys.exit(status)
