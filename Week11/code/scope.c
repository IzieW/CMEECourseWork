#include <stdio.h>

int y;
int x = 1;

int main(void)
{
    int x = 4;

    {
        int x = 5;
    }

    printf("The value of x: %i\n", x);  // x will take highest level of scope
    printf("The value of y: %i\n", y);

    return 0;
}