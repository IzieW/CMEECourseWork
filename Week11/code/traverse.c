#include <stdio.h>

int main(void)
{
    int i;
    int numbers[] = {1, 2, 3, 4, 5};
    // Increment operator
    i++;
    // Deincrement
    i--; // or --i;

    // Print elements forward:
    for (i = 0; i < 5; ++i) {
        printf("%i", numbers[i]);
    }
    printf("\n");

    // Print in reverse
    for(i = 4; i >= 0; i--){
        printf("%i", numbers[i]);
    }
    printf("\n");

    // Can also do this: 
    for (i = 5; i--; ){  // Can do this since each loop it will first check for i, then deincrement it
        printf("%i ", numbers[i]);
    }
    printf("\n");

    return 0;
}