#include <stdio.h>
#include <stdlib.h>

void doubler(int* i)
{
    // i = i * 2
    // return i;

    // Express this using pointer syntax 

    *i = *i * 2;
}

int main(void)
{
    int x;
    int* xp;

    int integs[] = {1, 2, 3, 4, 5};
    
    int *z;
    z = &integs[0];
    z = integs;  // two expressions are the same: C cannot point to whole arrays, so 
    // When pointing to whole array, actually only pointing to first integer in the array.

    printf("the third element of the array: %i\n", *(z + 2)); 
    printf("the third element of the array: %i\n", z[2]);  // two expressions also the same!

    xp = NULL;

    xp = &x;

    printf("x before initilisation: %i\n", x);  // Garbage value

    *xp = 7;

    printf("x after initialisation: %i\n", x);

    printf("and *xp is: %i\n", *xp);  // Works both ways. 

    // meanwhile...

    printf("xp still equals: %ls\n", xp);  // can't be printed, but &x is the answer. The addres. 

    doubler(xp);

    printf("x is now: %i\n", x);

    return 0;
}