#include <stdio.h>

int main(void)
{
    char i; 
    int x;

    i = 9;  // No warnings for writing an integer into char i
    x = 256;
    printf("The value is c: %i\n", i);  // Works fine because told to interpret as an integer
    printf("The value in x: %i\n", x);

    i = x;
    printf("This one will produce a warning: %i\n", i);  //  Returns 0 as max character is 255, so defaults to zero
    //  if 257 is passed in, it will return 1, as cycles back through. 
    
    return 0;
}