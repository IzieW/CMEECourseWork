#include <stdio.h>

int main(void)
{
    // Write a function that finds the minimum value in the set of numbers

    int set[] = {123, 747, 768, 2742, 988, 1121, 109, 999, 727, 1030, 999, 2014, 1402};

    int i;
    int min = 3000;
    for (i=0; i<13; i++) {
        if (set[i] < min) {
            min = set[i];
        }
    }

    printf("The minimum value is : %i\n", min);


}