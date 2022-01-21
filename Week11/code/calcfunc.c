#include <stdio.h>

float divide ( float num, float demon);  // Function prototype: put on top so compiler knows meaning of division as it reads main argument
// Alternatively, can put main function at the bottom of script

int main(void)
{
    int x, y;
    float f;

    x = 7;
    y = 3;

    f = x / y; 
    
    printf("The result of int division: %f\n", f);

    f = divide(x, y);

    printf("The result of division using a function: %f\n", f);


    return 0;
}

float divide (float num, float denom)
{
    float res;  // Automatic local variable
    res = num / denom;

    return res;
}