#include <stdio.h>

int main(void)
{
    int a = 7;
    int b = 2;
    float c = 0;

    c = (float) a/b;  // "Typecasting" - specifying a is a float. Variable is converted to float before division
    // Typecast operator has higher precedence than the divide operator.
    // So a is float, divided by integer. Not a/b is float. 

    printf("%f\n", c);
}