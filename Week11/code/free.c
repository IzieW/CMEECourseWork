#include <stdio.h>
#include <stdlib.h>

/* Heap memory remains even after the program flow, so important to return it to the system
using free() at the end of the programme execution... otherwise it can slow
the whole machine down. 

Good practice to match any function that allocates memory with a function
that deallocates it using free().

Free() can be a bit tricky however... if a pointer is already freed, your system
will Crash, so recommended to use it in the following wrapper: */

int free_Safely(int* something_ptr)
{ // Takes a pointer to a pointer as its argument, safely frees and sets pointer to null
    if (something_ptr) {  // If pointer is not already NULL
        free(something_ptr);
        something_ptr = NULL; // reset pointer to null
    }
}

int main(void){
    int* i = NULL;
    i = (int*)malloc(20* sizeof(i));

    free_Safely(i);
    return 0;
}