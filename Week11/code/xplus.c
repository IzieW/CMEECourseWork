#include <stdio.h>

int main(void)
{
    int x = 1, y = 1;
    int rx, ry;

    rx = x++;  // rx = x; x = x + 1; Save x, then add 1
    ry = ++y;  // y = y + 1; ry = y; Add one, then save x

    printf("x is: %i\n", rx);  // 1
    printf("y is: %i\n", ry);  // 2

    return 0;
}