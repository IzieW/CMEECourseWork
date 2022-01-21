#include <stdio.h>

int main(void)
{
    int x;
   int myarray1[5];  // Explicit definition/sizing of array
   int myarray2[ ] = {7, 9, 21, 55, 199191, 4, 18};  // Size is implicit in contents

    printf("%li\n", sizeof(myarray2)); // Prints size of array in bites (28 bites)
    // 1 integer = 4 bites

    // initialise myarray1:
     x = myarray1[2];

     for (x = 0; x < 5; ++x){
         myarray1[x] = 0;
     }

    printf("x is a value from outside the array: %i\n", myarray1[5]);  // This is a BUG!

    return 0;
}