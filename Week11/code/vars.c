#include <stdio.h>

int main(void)
{
    int x;  // x is a variable
    x = 5;  // 0 is a constant literal 

    printf("The value of x is: %i\n", x);  // %i allows print of variable


    float y = 1.2; 

    // Integral:

    // 0000 : 0
    // 0001 : 1
    // 0010: 2

    char c;  // declare a character 1-byte
    int i; // 4-bytes (integral), 32-bit
    long int li;
    long long int lli; 

    // Floating-point
    float f;
    double d;
    long double ld;

    // Basic operators:
    // +, -, *, /, % (modulo)
   
    x = 1.2;

    printf("The value of x is: %i\n", x);  // truncates to make x an integer (1)


    return x;
}